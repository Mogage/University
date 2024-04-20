import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

from dataset import Dataset
from train import train_model
from evaluate import evaluate_model
from model import Classifier


# def plot_data(dataset):
#     values = []
#     for i in range(len(dataset)):
#         values.append(dataset[i][0].numpy())
#
#     plt.plot(values)
#     plt.show()
#
#
# def train():
#     train_dataset = TrainDataset()
#     val_dataset = ValDataset()
#     plot_data(train_dataset)
#     plot_data(val_dataset)
#
#     train_size = len(train_dataset)
#
#     batch_size = 32
#     learning_rate = 0.001
#     epochs = 50
#
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     model = Classifier(35, 18, 9)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
#     model = train_model(model,
#                         criterion,
#                         optimizer,
#                         train_dataloader,
#                         val_dataloader,
#                         device,
#                         train_size,
#                         epochs)
#
#     model.save(f'model_{evaluate_model(model, val_dataloader, device)*100:.2f}.pt')
#
#
# def predict():
#     test_dataset = TestDataset()
#     test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = Classifier(35, 15, 9)
#     model.load(f'model_69.80.pt')
#     model.to(device)
#     model.eval()
#     data = {"Label": []}
#     with torch.no_grad():
#         for inputs, labels in test_dataloader:
#             inputs = inputs.to(device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             data["Label"].append(predicted.item())
#     df = pd.DataFrame(data)
#     df.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    dataset = Dataset()
    print(dataset[0][0]['Subject'])
    print(dataset[0][0]['From'])