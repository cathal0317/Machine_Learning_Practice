import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import matplotlib as mpl


# Define Hyperparameters
num_epochs = 10
lr = 1e-3
batch_size = 10

# Data
# Download Data
mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [0.8, 0.2])

# Create Dataloader
dataloader = {}
dataloader['train'] = DataLoader(mnist_train, batch_size=batch_size, shuffle = True)
dataloader['test'] = DataLoader(mnist_test, batch_size=batch_size, shuffle = True)
dataloader['val'] = DataLoader(mnist_val, batch_size=batch_size, shuffle = True)

print(len(dataloader["train"]))

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(256, 10)
        self.fc_var = nn.Linear(256, 10)

        self.decoder = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        recon = self.decoder(z)
        return recon
    
    def forward(self, x):
        batch_size = x.size(0)
        mu, log_var = self.encode(x.view(batch_size, -1))
        z = self.reparameterize(mu, log_var)
        out = self.decode(z)
        return out, mu, log_var
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
MSE = torch.nn.MSELoss(reduction='sum')

def loss_func(x, recon_x, mu, log_var):
    MSE_loss = MSE(recon_x, x.view(-1, 784))
    KL_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return MSE_loss + KL_loss

model = VAE().to(device)

optimiser = torch.optim.Adam(mdoel.parameter(), lr = learning_rate)
