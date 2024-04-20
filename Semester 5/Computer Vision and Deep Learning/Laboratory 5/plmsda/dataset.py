import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import email


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.train_data_folder = "data/train/"
        self.file_names = os.listdir(self.train_data_folder)
        df = pd.read_csv("train.csv")
        self.targets = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        with open(os.path.join(self.train_data_folder, self.file_names[idx]), "rb") as f:
            email_bytes = f.read()
            input_msg = email.message_from_bytes(email_bytes)

        return input_msg, self.targets[idx]
#
# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self, file_path):
#         super().__init__()
#         df = pd.read_csv(file_path)
#         df = df.drop(columns=['is_sm_ips_ports', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd'], axis=1)
#         self.features = df.iloc[:, :-1].fillna(0)
#         numerical_features = self.features.select_dtypes(include=['float64', 'int64']).columns
#         self.numerical_data = self.features[numerical_features]
#         scaler = StandardScaler()
#         self.numerical_data = torch.tensor(scaler.fit_transform(self.numerical_data), dtype=torch.float32)
#
#         categorical_features = self.features.select_dtypes(include=['object']).columns
#         self.categorical_data = self.features[categorical_features]
#         self.categorical_data = self.categorical_data.apply(LabelEncoder().fit_transform)
#         self.categorical_data = torch.tensor(self.categorical_data.values, dtype=torch.long)
#
#         self.inputs = torch.cat((self.numerical_data, self.categorical_data), dim=1)
#         self.targets = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)
#
#     def __len__(self):
#         return len(self.inputs)
#
#     def __getitem__(self, idx):
#         return self.inputs[idx], self.targets[idx]
#
#
# class TrainDataset(CustomDataset):
#     def __init__(self):
#         super().__init__('train.csv')
#
#
# class ValDataset(CustomDataset):
#     def __init__(self):
#         super().__init__('val.csv')
#
#
# class TestDataset(torch.utils.data.Dataset):
#     def __init__(self):
#         super().__init__()
#         df = pd.read_csv("test_features.csv")
#         df = df.drop(columns=['is_sm_ips_ports', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd'], axis=1)
#         self.features = df.iloc[:, :].values
#         numerical_features = self.features.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
#         self.numerical_data = self.features[numerical_features]
#         scaler = StandardScaler()
#         self.numerical_data = torch.tensor(scaler.fit_transform(self.numerical_data), dtype=torch.float32)
#
#         categorical_features = self.features.select_dtypes(include=['object']).columns
#         self.categorical_data = self.features[categorical_features]
#         self.categorical_data = self.categorical_data.apply(LabelEncoder().fit_transform)
#         self.categorical_data = torch.tensor(self.categorical_data.values, dtype=torch.long)
#
#         self.inputs = torch.cat((self.numerical_data, self.categorical_data), dim=1)
#         self.targets = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)
#
#     def __len__(self):
#         return len(self.inputs)
#
#     def __getitem__(self, idx):
#         return self.inputs[idx], self.targets[idx]
