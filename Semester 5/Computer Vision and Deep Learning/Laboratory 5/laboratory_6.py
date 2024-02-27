import os

import numpy as np
import torch
import gradio as gr
from torchvision.transforms import v2

from net.net_model import Net


def save_scripted_model():
    model = Net(n_channels=3, n_classes=3)
    if os.path.exists(f'models/{dataset_name}_{epochs}_model.pt'):
        model.load(f'models/{dataset_name}_{epochs}_model.pt')

    model.eval()
    scripted_model = torch.jit.script(model)

    scripted_model.save(f'models/{dataset_name}_{epochs}_scripted_model.pt')


def segment_image(image):
    model = torch.jit.load(f'models/{dataset_name}_{epochs}_scripted_model.pt')

    transform = v2.Compose([
        v2.ToTensor(),
        # v2.Resize((256, 256))
    ])
    image = transform(image)
    image = image.unsqueeze(0)

    mask = model(image)
    mask = mask[0].detach().cpu().numpy()
    mask = mask.transpose(1, 2, 0)
    mean = np.mean(mask)
    std = np.std(mask)

    mask -= mean
    mask /= std
    mask = mask.clip(-1, 1)
    return mask


if __name__ == '__main__':
    epochs = 60
    dataset_name = 'lfw'

    # save_scripted_model()

    ui = gr.Interface(
        fn=segment_image,
        inputs=gr.Image(),
        outputs=gr.Image(),
        title="Semantic Segmentation",
        description="Segmentation of faces"
    )

    ui.launch()
