import torch
from torchvision.models import resnet101

if __name__ == '__main__':
    resnet101 = resnet101(pretrained=True)
    torch.save(resnet101, 'resnet101.pth')

