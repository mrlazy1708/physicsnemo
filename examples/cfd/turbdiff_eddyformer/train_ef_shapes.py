import hydra
from tqdm import tqdm

from typing import Tuple
from torch import Tensor
from omegaconf import DictConfig

import os
import collections
import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel

import wandb
from omegaconf import OmegaConf

from physicsnemo.models.eddyformer import EddyFormer, EddyFormerConfig
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from physicsnemo.launch.utils import save_checkpoint
from physicsnemo.launch.logging import PythonLogger, LaunchLogger


def MSE(pred: Tensor, target: Tensor) -> Tensor:
    return torch.mean((pred - target) ** 2)

class TurbDiff(Dataset):

    path: str
    t: float

    num_samples: int = 5000
    samples_per_file: int = 100

    def __init__(self, root: str, split: str) -> None:
        """
        """
        super().__init__()
        self.path = f"{root}/{split}"

        self.shapes = []
        for dir in sorted(os.listdir(self.path)):
            self.shapes.append(dir)

    def __len__(self) -> int:
        return len(self.shapes) * (self.num_samples - 1)

    def load_snapshot(self, shape_idx: int, time_idx: int, test: bool = False) -> Tensor:
        chunk_idx, time_idx = divmod(time_idx, self.samples_per_file)
        data = np.load(f"{self.path}/{self.shapes[shape_idx]}/data{chunk_idx+1:03d}.npy", allow_pickle=True)

        mean = torch.tensor([ 1.2742623e+01,  7.3786831e-04,  9.0422938e-03, -5.4191113e+01])
        std = torch.tensor([ 11.82508  ,   4.8322463,   4.913811 , 219.8864   ])

        sample = (torch.from_numpy(data.item()["data_3d"][time_idx]) - mean) / std
        if test:

            mask = torch.from_numpy(data.item()["inside_mask"])[..., None]
            sample = torch.concat([sample, mask], dim=-1)
            print(sample.shape)

        return sample

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        shape_idx, time_idx = divmod(idx, self.num_samples - 1)

        return self.load_snapshot(shape_idx, time_idx, test=True), \
               self.load_snapshot(shape_idx, time_idx + 1)

    def metric(self, pred: Tensor, target: Tensor) -> dict[str, float]:
        """
        """
        mse = [MSE(pred[..., i], target[..., i]).item() for i in range(4)]
        return { f"mse_{ax}": value for ax, value in (zip("xyzp", mse)) }

@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def shapes_trainer(cfg: DictConfig) -> None:
    """
    """
    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    # initialize monitoring
    log = PythonLogger(name="shapes_ef")
    os.makedirs(cfg.training.result_dir, exist_ok=True)
    log.file_logging(f"{cfg.training.result_dir}/log.txt")
    LaunchLogger.initialize()  # PhysicsNeMo launch logger
    
    if dist.rank == 0:
        # init wandb
        wandb.init(
            project="physicsnemo",
            name=f"shapes_ef_{cfg.training.result_dir}",
            group=f"shapes_ef",
            config=OmegaConf.to_container(cfg, resolve=True),
            mode="online",
            # resume="must",
        )

    # define model and optimizer
    model = EddyFormer(
        idim=cfg.model.idim,
        odim=cfg.model.odim,
        hdim=cfg.model.hdim,
        num_layers=cfg.model.num_layers,
        use_scale=cfg.model.use_scale,
        cfg=EddyFormerConfig(
            basis=cfg.model.layer_config.basis,
            mesh=tuple(cfg.model.layer_config.mesh),
            mode=tuple(cfg.model.layer_config.mode),
            mode_les=tuple(cfg.model.layer_config.mode_les),
            kernel_size=tuple(cfg.model.layer_config.kernel_size),
            kernel_size_les=tuple(cfg.model.layer_config.kernel_size_les),
            ffn_dim=cfg.model.layer_config.ffn_dim,
            activation=cfg.model.layer_config.activation,
            num_heads=cfg.model.layer_config.num_heads,
            heads_dim=cfg.model.layer_config.heads_dim,
        ),
    ).to(dist.device)

    if dist.distributed:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            model = DistributedDataParallel(
                model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)
        log.success("Initialized DDP training with number of processes: {dist.world_size}")

    optimizer = Adam(model.parameters(), lr=cfg.training.learning_rate)

    # define dataset and dataloader
    dataset = TurbDiff(root=cfg.training.dataset, split="train")
    dataloader = DataLoader(dataset, cfg.training.batch_size, shuffle=True)

    testset = TurbDiff(root=cfg.training.dataset, split="test")
    testloader = DataLoader(testset, batch_size=None)
     
    # define training step
    @StaticCaptureTraining(
        model=model,
        optim=optimizer,
        logger=log,
        use_graphs=False,
        use_amp=cfg.training.amp,
        compile=cfg.training.compile
    )
    def training_step(input: Tensor, target: Tensor) -> Tensor:
        pred = torch.vmap(model)(input)
        loss = torch.vmap(MSE)(pred, target)
        return torch.mean(loss)

    # define evaluation step
    @StaticCaptureEvaluateNoGrad(
        model=model,
        logger=log,
        use_graphs=False,
        use_amp=cfg.training.amp,
        compile=cfg.training.compile
    )
    def forward_eval(input):
        return model(input)

    it = 0

    model.train()
    log.info("Training started")

    for epoch in range(cfg.training.num_epochs):
        for it, (input, target) in tqdm(enumerate(dataloader), desc="Training", total=len(dataloader), leave=False, unit="batch"):

            input = input.to(dist.device)
            target = target.to(dist.device)
            loss = training_step(input, target)

            if dist.rank == 0:
                # with LaunchLogger("train", epoch=epoch) as logger:
                # logger.log_minibatch({"Training loss": loss.item()})
                wandb.log({"Training loss": loss.item()})

            if it and it % cfg.training.ckpt_every == 0 and dist.rank == 0:
                save_checkpoint(f"{cfg.training.result_dir}/ckpt.pt", model, optimizer, epoch=it)

            # if it and it % cfg.training.test_every == 0:

            #     model.eval()
            #     metrics = collections.defaultdict(float)

            #     for input, target in tqdm(testloader, desc="Test"):

            #         input = input.to(dist.device)
            #         target = target.to(dist.device)

            #         pred = forward_eval(input)
            #         metric = testset.metric(pred, target)

            #         for key, value in metric.items():
            #             metrics[key] += value / len(testset)

            #     with LaunchLogger("test", epoch=epoch) as logger:
            #         logger.log_minibatch(metrics)

            #     model.train()

    log.success("Training completed")
    save_checkpoint(f"{cfg.training.result_dir}/ckpt.pt", model, optimizer)


if __name__ == "__main__":
    shapes_trainer()
