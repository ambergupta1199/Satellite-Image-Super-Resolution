import os

# from datetime import datetime
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn

# import numpy as np
# import pandas as pd
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Local imports
from configs import Config
from dataset import SatelliteDataset
from models.srgan_components import Discriminator, Generator


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


class Trainer:
    def __init__(self, rank, world_size, config):
        self.rank = rank
        self.world_size = world_size
        self.config = config

        # Initialize models
        self.generator = Generator(config).to(rank)
        self.discriminator = Discriminator().to(rank)

        # Distributed Data Parallel
        self.generator = DDP(self.generator, device_ids=[rank])
        self.discriminator = DDP(self.discriminator, device_ids=[rank])

        # Optimizers
        self.opt_g = torch.optim.AdamW(
            self.generator.parameters(), lr=config.lr, betas=config.betas
        )
        self.opt_d = torch.optim.AdamW(
            self.discriminator.parameters(), lr=config.lr, betas=config.betas
        )

        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        # Metrics
        self.ssim = StructuralSimilarityIndexMeasure().to(rank)
        self.psnr = PeakSignalNoiseRatio().to(rank)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(rank)
        self.fid = FrechetInceptionDistance().to(rank)

        # VGG for perceptual loss
        self.vgg = (
            nn.Sequential(
                *list(
                    torch.hub.load("pytorch/vision", "vgg19", pretrained=True).features[
                        :16
                    ]
                )
            )
            .eval()
            .to(rank)
        )

        # Dataset and loader
        self.dataset = SatelliteDataset(config)
        self.sampler = DistributedSampler(
            self.dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            sampler=self.sampler,
            num_workers=config.num_workers,
            pin_memory=True,
        )

    def _calculate_perceptual_loss(self, fake, real):
        with torch.no_grad():
            real_features = self.vgg(real)
        fake_features = self.vgg(fake)
        return self.l1_loss(fake_features, real_features)

    def _train_generator(self, lr, hr, text_emb):
        self.opt_g.zero_grad()

        # Generate HR image
        fake_hr = self.generator(lr, text_emb)

        # Calculate losses
        pixel_loss = self.l1_loss(fake_hr, hr)
        perc_loss = self._calculate_perceptual_loss(fake_hr, hr)
        adv_loss = -torch.mean(self.discriminator(fake_hr))

        # Total loss
        total_loss = (
            self.config.λ_pixel * pixel_loss
            + self.config.λ_perc * perc_loss
            + self.config.λ_adv * adv_loss
        )

        total_loss.backward()
        self.opt_g.step()

        return {
            "g_loss": total_loss.item(),
            "pixel_loss": pixel_loss.item(),
            "perc_loss": perc_loss.item(),
            "adv_loss": adv_loss.item(),
        }

    def _train_discriminator(self, fake_hr, real_hr):
        self.opt_d.zero_grad()

        # Real images
        pred_real = self.discriminator(real_hr)
        loss_real = -torch.mean(pred_real)

        # Fake images
        pred_fake = self.discriminator(fake_hr.detach())
        loss_fake = torch.mean(pred_fake)

        # Total loss
        total_loss = loss_real + loss_fake
        total_loss.backward()
        self.opt_d.step()

        return {
            "d_loss": total_loss.item(),
            "real_loss": loss_real.item(),
            "fake_loss": loss_fake.item(),
        }

    def _validate(self):
        self.generator.eval()
        metrics = defaultdict(float)
        self.fid.reset()

        with torch.no_grad():
            for lr, hr, text in self.loader:
                lr = lr.to(self.rank)
                hr = hr.to(self.rank)
                text = text.to(self.rank)

                fake_hr = self.generator(lr, text)

                # Update metrics
                metrics["ssim"] += self.ssim(fake_hr, hr).item()
                metrics["psnr"] += self.psnr(fake_hr, hr).item()
                metrics["lpips"] += self.lpips(fake_hr, hr).item()
                metrics["mse"] += self.mse_loss(fake_hr, hr).item()
                metrics["mae"] += self.l1_loss(fake_hr, hr).item()

                # Update FID
                real_uint8 = (hr * 255).byte()
                fake_uint8 = (fake_hr * 255).byte()
                self.fid.update(real_uint8, real=True)
                self.fid.update(fake_uint8, real=False)

        # Average metrics
        for k in metrics:
            metrics[k] /= len(self.loader)
        metrics["fid"] = self.fid.compute().item()

        return metrics

    def train(self):
        for epoch in range(self.config.epochs):
            self.sampler.set_epoch(epoch)
            self.generator.train()
            self.discriminator.train()

            epoch_metrics = defaultdict(float)

            for batch_idx, (lr, hr, text) in enumerate(self.loader):
                lr = lr.to(self.rank, non_blocking=True)
                hr = hr.to(self.rank, non_blocking=True)
                text = text.to(self.rank, non_blocking=True)

                # Train generator
                g_metrics = self._train_generator(lr, hr, text)

                # Train discriminator
                with torch.no_grad():
                    fake_hr = self.generator(lr, text)
                d_metrics = self._train_discriminator(fake_hr, hr)

                # Aggregate metrics
                for k, v in g_metrics.items():
                    epoch_metrics[k] += v
                for k, v in d_metrics.items():
                    epoch_metrics[k] += v

            # Log training metrics
            if self.rank == 0:
                log_str = f"Epoch {epoch + 1}/{self.config.epochs} | "
                for k, v in epoch_metrics.items():
                    log_str += f"{k}: {v / len(self.loader):.4f} | "
                print(log_str)

            # Validation
            if (epoch + 1) % self.config.val_interval == 0 and self.rank == 0:
                val_metrics = self._validate()
                log_str = f"Validation | Epoch {epoch + 1} | "
                for k, v in val_metrics.items():
                    log_str += f"{k}: {v:.4f} | "
                print(log_str)

                # Save checkpoint
                torch.save(
                    {
                        "epoch": epoch,
                        "generator": self.generator.module.state_dict(),
                        "discriminator": self.discriminator.module.state_dict(),
                        "opt_g": self.opt_g.state_dict(),
                        "opt_d": self.opt_d.state_dict(),
                    },
                    f"{self.config.checkpoint_dir}/checkpoint_{epoch + 1}.pth",
                )


def main(rank, world_size):
    setup(rank, world_size)
    config = Config()

    # Create checkpoint directory
    if rank == 0 and not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    trainer = Trainer(rank, world_size, config)
    trainer.train()

    cleanup()


if __name__ == "__main__":
    config = Config()
    world_size = torch.cuda.device_count()

    # Set deterministic flags
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)

    # Start distributed training
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
