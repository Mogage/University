class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 7)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.max_pool2d(torch.relu(self.bn1(self.conv1(x))), kernel_size=2, stride=2)
        x = torch.max_pool2d(torch.relu(self.bn2(self.conv2(x))), kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        emotion_output = torch.softmax(self.fc2(x), dim=1)
        arousal_output = self.fc3(x)
        return emotion_output, arousal_output
