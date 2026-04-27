import torch
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

from torchvision import datasets
from torchvision.transforms import ToTensor


train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)

# train_data.data are the X_i's
print(train_data.data.size())

# train_data.targets are the Y_i's
print(train_data.targets.size())

import matplotlib.pyplot as plt
k = 10
plt.imshow(train_data.data[k])
print('Label: ',train_data.targets[k])      

import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x  # return x for visualization
    cnn = CNN()
    print(cnn)``