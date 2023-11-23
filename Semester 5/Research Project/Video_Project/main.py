import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from torch_enhance.datasets import BSDS300, Set14, Set5
from torch_enhance.models import SRResNet
from torch_enhance import metrics

from skimage.metrics import structural_similarity as ssim


class Module(pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self(lr)
        loss = F.mse_loss(sr, hr, reduction="mean")

        # metrics
        mae = metrics.mae(sr, hr)
        psnr = metrics.psnr(sr, hr)

        hr_np = hr.detach().cpu().numpy().transpose(0, 2, 3, 1)
        sr_np = sr.detach().cpu().numpy().transpose(0, 2, 3, 1)

        ssim_value = np.mean(np.array([
            ssim(hr_i, sr_i, multichannel=True, gaussian_weights=True, sigma=1.5,
                 use_sample_covariance=False, win_size=3,
                 data_range=sr_i.max() - sr_i.min())
            for hr_i, sr_i in zip(hr_np, sr_np)
        ]))

        # Logs
        self.log("train_loss", loss)
        self.log("train_mae", mae)
        self.log("train_psnr", psnr)
        self.log("train_ssim", ssim_value)

        return loss

    def validation_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self(lr)
        loss = F.mse_loss(sr, hr, reduction="mean")

        # metrics
        mae = metrics.mae(sr, hr)
        psnr = metrics.psnr(sr, hr)

        # Logs
        hr_np = hr.detach().cpu().numpy().transpose(0, 2, 3, 1)
        sr_np = sr.detach().cpu().numpy().transpose(0, 2, 3, 1)

        ssim_value = np.mean(np.array([
            ssim(hr_i, sr_i, multichannel=True, gaussian_weights=True, sigma=1.5,
                 use_sample_covariance=False, win_size=3,
                 data_range=sr_i.max() - sr_i.min())
            for hr_i, sr_i in zip(hr_np, sr_np)
        ]))

        # Logs
        self.log("train_loss", loss)
        self.log("train_mae", mae)
        self.log("train_psnr", psnr)
        self.log("train_ssim", ssim_value)

        return loss

    def test_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self(lr)
        loss = F.mse_loss(sr, hr, reduction="mean")

        # metrics
        mae = metrics.mae(sr, hr)
        psnr = metrics.psnr(sr, hr)

        # Logs
        hr_np = hr.detach().cpu().numpy().transpose(0, 2, 3, 1)
        sr_np = sr.detach().cpu().numpy().transpose(0, 2, 3, 1)

        ssim_value = np.mean(np.array([
            ssim(hr_i, sr_i, multichannel=True, gaussian_weights=True, sigma=1.5,
                 use_sample_covariance=False, win_size=3,
                 data_range=sr_i.max() - sr_i.min())
            for hr_i, sr_i in zip(hr_np, sr_np)
        ]))

        # Logs
        self.log("train_loss", loss)
        self.log("train_mae", mae)
        self.log("train_psnr", psnr)
        self.log("train_ssim", ssim_value)

        return loss


scale_factor = 4

# Setup dataloaders
train_dataset = BSDS300(scale_factor=scale_factor)
val_dataset = Set14(scale_factor=scale_factor)
test_dataset = Set5(scale_factor=scale_factor)

# train_dataset = MyDataset()

# train_dataloader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1)
test_dataloader = DataLoader(test_dataset, batch_size=1)

# Define model
channels = 3
model = SRResNet(scale_factor, channels)
module = Module(model)

trainer = pl.Trainer(max_epochs=50)
trainer.fit(
    module,
    train_dataloader,
    val_dataloader,
)
module.eval()
trainer.test(module, test_dataloader)
