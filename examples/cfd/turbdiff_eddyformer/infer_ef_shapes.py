import hydra
from tqdm import tqdm, trange
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from physicsnemo.models.eddyformer import EddyFormer, EddyFormerConfig
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.utils import load_checkpoint
from physicsnemo.launch.logging import PythonLogger, LaunchLogger
from omegaconf import DictConfig
from typing import Tuple
from torch import Tensor
from itertools import islice

# Import Re94 and rel_l2 from the training script
import sys
sys.path.append(os.path.dirname(__file__))
from train_ef_shapes import TurbDiff

class TurbDiff_testset(torch.utils.data.Dataset):

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
        self.time_idx = 0
        for dir in sorted(os.listdir(self.path)):
            self.shapes.append(dir)

    def __len__(self) -> int:
        return len(self.shapes)

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
        # shape_idx, time_idx = divmod(idx, self.num_samples - 1)
        shape_idx = idx
        return self.shapes[shape_idx], self.load_snapshot(shape_idx, self.time_idx, test=True), \
               self.load_snapshot(shape_idx, self.time_idx + 1)




@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def shapes_infer(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    log = PythonLogger(name="shapes_ef_infer")
    LaunchLogger.initialize()
    print("cfg: ", cfg)
    # define model
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

    # Load checkpoint
    ckpt_dir = cfg.inference.ckpt_dir
    epoch = cfg.inference.epoch
    
    log.info(f"Loading checkpoint from {ckpt_dir} at epoch {epoch}")
    load_checkpoint(ckpt_dir, models=model, epoch=epoch, device=dist.device)
    model.eval()

    # define dataset and dataloader
    # Use split="test" for inference
    testset = TurbDiff_testset(root=cfg.training.dataset, split="test")
    testloader = DataLoader(testset, batch_size=None)

    preds = []
    targets = []
    shapes = []
    rollout_steps = 10
    log.info(f"Starting inference on {len(testset)} samples")
    with torch.no_grad():
        pred = None
        for n in tqdm(range(len(testset))):
            shape, input, target = testset[n]
            print(input.shape, target.shape)
            input = input.to(dist.device)

            if pred is not None: # update input with prediction
                mask = input[..., -1:]
                pred *= mask
                input = torch.concat([pred, mask], dim=-1)

            # EddyFormer expects (C, H, W, D) if no batch dim or (B, C, ...)
            # training uses torch.vmap(model)(input) where input is (B, C, H, W, D)
            # evaluation uses model(input)
            for step in rollout_steps:
                pred = model(input)
                preds.append(pred.cpu().numpy())
                targets.append(target.numpy())
                shapes.append(shape)
                # break # only run on one sample

    
    
    output_dir = cfg.inference.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_epoch_{epoch}.npy")
    np.save(output_path, {"pred": preds, "target": targets, "shapes": shapes})
    log.success(f"Saved predictions to {output_path}")

if __name__ == "__main__":
    shapes_infer()
