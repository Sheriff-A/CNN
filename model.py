import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, in_channels, output):
        super(CNN, self).__init__()
        # Input Channels = 3, defined by the data
        # Output Channels = 20
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)

        # Input Channels = 20
        # Output Channels = 15
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=15, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Input Features = 15 * 15 * 15
        # Output Features = 100
        self.conv2linear_input_shape = 15 * 15 * 15
        self.fc1 = nn.Linear(self.conv2linear_input_shape, 256)

        # Input Features = 100
        # Output Features = 64
        self.fc2 = nn.Linear(256, 64)

        # Input Features = 64
        # Output Features = 10, defined by the data
        self.fc3 = nn.Linear(64, output)

    def forward(self, x):
        # Convolutional Feed Forward
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Fully Connected Linear Feed Forward
        x = x.view(-1, self.conv2linear_input_shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)
