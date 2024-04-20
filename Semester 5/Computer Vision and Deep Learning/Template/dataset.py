import os

import torch


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.images_path = 'images'
        self.masks_path = 'masks'
        self.transform = transform
        self.images = sorted(os.listdir(self.images_path))
        self.masks = sorted(os.listdir(self.masks_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_path, self.images[idx])
        mask_path = os.path.join(self.masks_path, self.masks[idx])
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask
