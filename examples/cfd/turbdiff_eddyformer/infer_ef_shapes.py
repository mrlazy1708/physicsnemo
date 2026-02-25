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
    ckpt_dir = "/global/homes/y/yuejian/project/Generative_3d_turbulence_flow/m4558/yihengdu/yihengdu/shapes_ef/ef-leg-fix/ckpt.pt"
    epoch = 20000
    
    log.info(f"Loading checkpoint from {ckpt_dir} at epoch {epoch}")
    load_checkpoint(ckpt_dir, models=model, epoch=epoch, device=dist.device)
    breakpoint()
    model.eval()

    # define dataset and dataloader
    # Use split="test" for inference
    testset = TurbDiff(root=cfg.training.dataset, split="test")
    testloader = DataLoader(testset, batch_size=None)

    preds = []
    targets = []
    
    log.info(f"Starting inference on {len(testset)} samples")
    with torch.no_grad():
        pred = None
        for n in tqdm(range(200, 300)):
            input, target = testset[n]
            input = input.to(dist.device)

            if pred is not None: # update input with prediction
                mask = input[..., -1:]
                pred *= mask
                input = torch.concat([pred, mask], dim=-1)

            # EddyFormer expects (C, H, W, D) if no batch dim or (B, C, ...)
            # training uses torch.vmap(model)(input) where input is (B, C, H, W, D)
            # evaluation uses model(input)
            pred = model(input)
                
            preds.append(pred.cpu().numpy())
            targets.append(target.numpy())

            # break # only run on one sample

    preds = np.stack(preds)
    targets = np.stack(targets)
    
    output_dir = "outputs/shapes/ef-leg-fix/inference"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_epoch_{epoch}.npy")
    np.save(output_path, {"pred": preds, "target": targets})
    log.success(f"Saved predictions to {output_path}")

if __name__ == "__main__":
    shapes_infer()
