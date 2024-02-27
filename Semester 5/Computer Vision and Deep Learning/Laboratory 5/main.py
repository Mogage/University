from dataset import LFWDataset, CelebADataset
from train import train_model
from evaluate import evaluate_model
from net.net_model import Net
import torch
import torchvision.transforms as v2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
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
    epochs = 10
    learning_rate = 0.001
    batch_size = 5
    patience = 4

    dataset = get_dataset(dataset_name)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = Net(n_channels=3, n_classes=3)
    if os.path.exists(f'models/{dataset_name}_50_model.pt'):
        model.load(f'models/{dataset_name}_50_model.pt')

    default_config = dict(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_percent=0.2,
        patience=patience,
        save_checkpoint=True
    )

    experiment = wandb.init(project='Segmentation', resume='allow', anonymous='must', config=default_config)

    logging.info(f'''Train starting with params:
                  Epochs: {epochs}
                  Batch size: {batch_size}
                  Learning rate: {learning_rate}
                  Val size: {val_size},
                  Device: {device},
          ''')

    ################## predict ##################
    # image = Image.open("lfw_dataset\\images\\Aaron_Peirsol\\Aaron_Peirsol_0001.jpg").convert('RGB')
    # # image = Image.open("CelebAMask-HQ\\images\\256.jpg").convert('RGB')
    #
    # transform = v2.Compose([
    #     v2.ToTensor()])
    # image = transform(image)
    # image = image.unsqueeze(0)
    #
    # model.eval()
    # mask = model(image)
    # # mask = F.interpolate(mask, size=(image.size(-2), image.size(-1)), mode='bilinear', align_corners=True)
    # # mask = mask.detach().cpu().numpy()
    # # mask = np.argmax(mask, axis=1)
    # mask = mask.squeeze(0).detach()
    #
    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(image.squeeze().permute(1, 2, 0))
    # axs[1].imshow(mask.permute(1, 2, 0))
    # plt.show()

    ################## train ##################
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

    model.save(f'models/{dataset_name}_60_model.pt')

    # ################## evaluate ##################
    # experiment = None
    # print(evaluate_model(model, test_dataloader, device, experiment))
