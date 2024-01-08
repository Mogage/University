from dataset import LFWDataset, CelebADataset
from train import train_model
from evaluate import evaluate_model
from net.net_model import Net
import torch
import torchvision.transforms as v2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import wandb
import logging


def get_dataset(dataset_type):
    transform = v2.Compose([
        v2.ToTensor()])
    if dataset_type == 'lfw':
        return LFWDataset(base_folder='lfw_dataset\\', transform=transform)
    elif dataset_type == 'celeb':
        return CelebADataset(base_folder='CelebAMask-HQ/', transform=transform)
    else:
        raise ValueError('Invalid dataset type')


if __name__ == "__main__":
    dataset_name = 'lfw'
    epochs = 50
    learning_rate = 0.001
    batch_size = 5

    dataset = get_dataset(dataset_name)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(n_channels=3, n_classes=3)
    if os.path.exists(f'{dataset_name}_{epochs}_model.pt'):
        model.load(f'{dataset_name}_{epochs}_model.pt')
    print(device)

    # experiment = wandb.init(project='Segmentation', resume='allow', anonymous='must')
    # experiment.config.update(
    #     dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, test_percent=0.2, save_checkpoint=False)
    # )
    #
    # logging.info(f'''Train starting with params:
    #               Epochs: {epochs}
    #               Batch size: {batch_size}
    #               Learning rate: {learning_rate}
    #               Test size: {test_size},
    #               Device: {device},
    #       ''')

    ################## predict ##################
    image = Image.open("lfw_dataset\\images\\Aaron_Peirsol\\Aaron_Peirsol_0001.jpg").convert('RGB')
    # image = Image.open("CelebAMask-HQ\\images\\256.jpg").convert('RGB')

    transform = v2.Compose([
        v2.ToTensor()])
    image = transform(image)
    image = image.unsqueeze(0)

    model.eval()
    mask = model(image)
    # mask = F.interpolate(mask, size=(image.size(-2), image.size(-1)), mode='bilinear', align_corners=True)
    # mask = mask.detach().cpu().numpy()
    # mask = np.argmax(mask, axis=1)
    mask = mask.squeeze(0).detach()

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image.squeeze().permute(1, 2, 0))
    axs[1].imshow(mask.permute(1, 2, 0))
    plt.show()

    ################## train ##################
    # criterion = nn.CrossEntropyLoss()
    # # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    #
    # train_model(model, criterion, optimizer, scheduler, train_dataloader, device, experiment, train_size, epochs)
    #
    # model.save(f'{dataset_name}_{epochs}_model.pt')

    # ################## evaluate ##################
    experiment = None
    print(evaluate_model(model, test_dataloader, device, experiment))
