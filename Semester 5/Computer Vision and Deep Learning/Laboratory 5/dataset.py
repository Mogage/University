import torch.utils.data
import os
from PIL import Image
import cv2
import glob


class LFWDataset(torch.utils.data.Dataset):
    def __init__(self, base_folder, transform):
        super().__init__()
        self.base_folder = base_folder

        self._transform = transform
        self.inputs = []
        self.targets = []
        self.__load_data()

    def __load_data(self):
        input_path = self.base_folder + 'images\\'
        target_path = self.base_folder + 'masks\\'

        for input_file_path in glob.glob(input_path + '**\\*.jpg', recursive=True):
            gt_path = target_path + input_file_path.split('\\')[3].split('.')[0] + '.ppm'
            if os.path.exists(gt_path):
                input_photo = Image.open(input_file_path).convert('RGB')
                gt_photo = Image.open(gt_path).convert('RGB')

                self.inputs.append(input_photo)
                self.targets.append(gt_photo)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        item = self.inputs[idx]
        target = self.targets[idx]

        if self._transform:
            item = self._transform(item)
            target = self._transform(target)

        return item, target


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, base_folder, transform=None):
        super().__init__()
        self.base_folder = base_folder
        self.images_path = base_folder + 'images'
        self.masks_path = base_folder + 'masks'
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
