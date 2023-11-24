import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl


class DoubleConvBlock(nn.Module):
    """double conv layers block"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downscale block: maxpool -> double conv block"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class BridgeDown(nn.Module):
    """Downscale bottleneck block: maxpool -> conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class BridgeUP(nn.Module):
    """Downscale bottleneck block: conv -> transpose conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.conv_up(x)


class UpBlock(nn.Module):
    """Upscale block: double conv block -> transpose conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConvBlock(in_channels * 2, in_channels)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return torch.relu(self.up(x))


class OutputBlock(nn.Module):
    """Output block: double conv block -> output conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_conv = nn.Sequential(
            DoubleConvBlock(in_channels * 2, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        return self.out_conv(x)


class ChannelReductionModel(nn.Module):
    def __init__(self):
        super(ChannelReductionModel, self).__init__()
        self.reduce_channels = nn.Conv2d(9, 3, kernel_size=1)

    def forward(self, x):
        return self.reduce_channels(x)


class DeepWBNet(nn.Module):
    def __init__(self):
        super(DeepWBNet, self).__init__()
        self.n_channels = 3
        self.encoder_inc = DoubleConvBlock(self.n_channels, 24)
        self.encoder_down1 = DownBlock(24, 48)
        self.encoder_down2 = DownBlock(48, 96)
        self.encoder_down3 = DownBlock(96, 192)
        self.encoder_bridge_down = BridgeDown(192, 384)
        self.awb_decoder_bridge_up = BridgeUP(384, 192)
        self.awb_decoder_up1 = UpBlock(192, 96)
        self.awb_decoder_up2 = UpBlock(96, 48)
        self.awb_decoder_up3 = UpBlock(48, 24)
        self.awb_decoder_out = OutputBlock(24, self.n_channels)
        self.tungsten_decoder_bridge_up = BridgeUP(384, 192)
        self.tungsten_decoder_up1 = UpBlock(192, 96)
        self.tungsten_decoder_up2 = UpBlock(96, 48)
        self.tungsten_decoder_up3 = UpBlock(48, 24)
        self.tungsten_decoder_out = OutputBlock(24, self.n_channels)
        self.shade_decoder_bridge_up = BridgeUP(384, 192)
        self.shade_decoder_up1 = UpBlock(192, 96)
        self.shade_decoder_up2 = UpBlock(96, 48)
        self.shade_decoder_up3 = UpBlock(48, 24)
        self.shade_decoder_out = OutputBlock(24, self.n_channels)

    def forward(self, x):
        x1 = self.encoder_inc(x)
        x2 = self.encoder_down1(x1)
        x3 = self.encoder_down2(x2)
        x4 = self.encoder_down3(x3)
        x5 = self.encoder_bridge_down(x4)
        x_awb = self.awb_decoder_bridge_up(x5)
        x_awb = self.awb_decoder_up1(x_awb, x4)
        x_awb = self.awb_decoder_up2(x_awb, x3)
        x_awb = self.awb_decoder_up3(x_awb, x2)
        awb = self.awb_decoder_out(x_awb, x1)
        x_t = self.tungsten_decoder_bridge_up(x5)
        x_t = self.tungsten_decoder_up1(x_t, x4)
        x_t = self.tungsten_decoder_up2(x_t, x3)
        x_t = self.tungsten_decoder_up3(x_t, x2)
        t = self.tungsten_decoder_out(x_t, x1)
        x_s = self.shade_decoder_bridge_up(x5)
        x_s = self.shade_decoder_up1(x_s, x4)
        x_s = self.shade_decoder_up2(x_s, x3)
        x_s = self.shade_decoder_up3(x_s, x2)
        s = self.shade_decoder_out(x_s, x1)
        return torch.cat((awb, t, s), dim=1)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.inputs = []
        self.targets = []
        self.transform = transform
        max_no_of_photos = 500
        index = 0
        image_size = (512, 512)
        for filename in os.listdir('Set2_input_images/'):
            img = cv2.imread('Set2_input_images/' + filename)
            img = cv2.resize(img, image_size)
            img = img.transpose((2, 0, 1)).astype(np.float32)
            self.inputs.append(img)
            index += 1
            if index == max_no_of_photos:
                break
        index = 0
        for filename in os.listdir('Set2_ground_truth_images/'):
            img = cv2.imread('Set2_ground_truth_images/' + filename)
            img = cv2.resize(img, image_size)
            img = img.transpose((2, 0, 1)).astype(np.float32)
            self.targets.append(img)
            index += 1
            if index == max_no_of_photos:
                break

        self.inputs = np.array(self.inputs)
        self.targets = np.array(self.targets)

        # self.inputs_train, self.inputs_test, self.targets_train, self.targets_test = train_test_split(
        #     self.inputs, self.targets, test_size=0.2)

        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        item = self.inputs[idx]
        target = self.targets[idx]

        if self.transform:
            item = self.transform(item)

        return item, target


class Module(pl.LightningModule):
    def __init__(self, model):
        super(Module, self).__init__()
        self.model = model
        self.reshaper = ChannelReductionModel()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        input_img, target = batch

        output = self(input_img)

        output = self.reshaper(output)

        loss = F.mse_loss(output, target)

        return loss


num_epochs = 5

dataset = MyDataset()
train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = DeepWBNet()

optimizer = optim.Adamax(model.parameters(), lr=1e-3)

# criterion = nn.MSELoss()
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#
#     for batch_idx, (inputs, targets) in enumerate(train_dataloader):
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#
#     average_loss = running_loss / len(train_dataloader)
#     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}")

trainer = pl.Trainer(max_epochs=num_epochs)
module = Module(model)
trainer.fit(module, train_dataloader)

reshaper_model = ChannelReductionModel()

model.eval()
total_ssim = 0.0

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_dataloader):
        outputs = model(inputs)
        outputs = reshaper_model(outputs)
        ssim_value = np.mean(np.array([
            ssim(targets_i, output_i, multichannel=True, gaussian_weights=True, sigma=1.5,
                 use_sample_covariance=False, win_size=3,
                 data_range=output_i.max() - output_i.min())
            for targets_i, output_i in zip(targets.cpu().numpy(), outputs.cpu().numpy())
        ]))
        total_ssim += ssim_value

average_ssim = total_ssim / len(test_dataloader)
print(f"Test SSIM: {average_ssim}")
