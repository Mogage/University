import os

import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader

import wandb
from main import get_dataset
from net.net_model import Net
from train import train_model


sweep_random_config = {
    "method": "random",
    "metric": {
        "name": "best_miou",
        "goal": "minimize"
    },
    "parameters": {
        "batch_size": {
            "min": 3,
            "max": 10
        },
        "learning_rate": {
            "min": 0.0001,
            "max": 0.1,
            "scale": "log"
        },
        "patience": {
            "min": 1,
            "max": 20
        }
    }
}

sweep_grid_config = {
    "method": "grid",
    "metric": {
        "name": "best_miou",
        "goal": "minimize"
    },
    "parameters": {
        "batch_size": {
            "values": [3, 4, 5, 6, 7, 8, 9, 10]
        },
        "learning_rate": {
            "values": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
        },
        "patience": {
            "values": [2, 3, 4, 5, 6, 7, 10, 15, 20]
        }
    }
}


def train(config=None):
    dataset_name = 'lfw'
    epochs = 2

    dataset = get_dataset(dataset_name)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    experiment = wandb.init(project='Segmentation', resume='allow', anonymous='must', config=config)

    config = experiment.config

    learning_rate = config.learning_rate
    batch_size = config.batch_size
    patience = config.patience

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(n_channels=3, n_classes=3)
    if os.path.exists(f'models/{dataset_name}_{epochs}_model.pt'):
        model.load(f'models/{dataset_name}_{epochs}_model.pt')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=patience)

    train_model(model,
                criterion,
                optimizer,
                scheduler,
                train_dataloader,
                val_dataloader,
                device,
                experiment,
                train_size,
                epochs)


if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep_random_config, project="Segmentation")
    wandb.agent(sweep_id, train, count=1)
    sweep_id = wandb.sweep(sweep_grid_config, project="Segmentation")
    wandb.agent(sweep_id, train, count=3)
