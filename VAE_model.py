import os
import sys
from abc import ABC

import torch
from torch import nn
import torch.nn.functional as F

def VAE_loss_binary(x, out, mu, logvar, beta):
    recons_loss = F.binary_cross_entropy(out, x)
    kld_loss = -1/2*torch.mean(torch.sum(1+logvar-torch.square(mu)-torch.exp(logvar), 1))
    loss = recons_loss + beta * kld_loss
    return recons_loss, kld_loss, loss

def VAE_loss_norm(x, out, mu, logvar, beta):
    recons_loss = 1/2*torch.mean((x-out)**2)
    kld_loss = -1/2*torch.mean(torch.sum(1+logvar-torch.square(mu)-torch.exp(logvar), 1))
    loss = recons_loss + beta * kld_loss
    return recons_loss, kld_loss, loss

class VAE(nn.Module):
    """
    __init__ must initialize self.encoder, self.decoder, self.mu, self.logvar
    and self.device
    """
    def encode(self, input):
        input=self.encoder(input)
        mu=self.mu(input)
        logvar=self.logvar(input)

        return mu, logvar

    def sample(self, mu, logvar):
        eps=torch.randn(mu.shape).to(self.device)
        std=torch.exp(0.5*logvar)
        return eps * std + mu

    def decode(self, input):
        out=self.decoder(input)
        return out

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.sample(mu, logvar)
        out = self.decode(z)
        return mu, logvar, out

    def generate(self, n):
        z = torch.randn(n, self.latent_dim).to(self.device)
        samples = self.decode(z)
        return samples

#Original model I had built for class
class ConvVAE(VAE):
    def __init__(self, device, latent_dim, *args, **kwargs):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        self.device = device

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 16*128),
            nn.ReLU(),
            nn.BatchNorm1d(16*128),
            nn.Unflatten(1, (128,4, 4)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1),

        )

        self.mu= nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*128, self.latent_dim),
            nn.ReLU()
        )

        self.logvar=nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*128, self.latent_dim),
            nn.ReLU()
        )

class ResBlock(nn.Module):
    def __init__(self, channels, kernal_size):
        super(ResBlock, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, channels, kernal_size, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernal_size, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.ReLU = nn.ReLU()
        self.Norm = nn.BatchNorm2d(channels)

    def forward(self, input):
        x = input + self.encoder(input)
        x = self.ReLU(x)
        return x

class ResVAE(VAE):
    """
    This a VAE based on the ResNET architecture. This model only works on inputs that are square whose lengths are a power of 2
    """
    def __init__(self, device, input_size: int, input_channels: int, latent_dim: int, num_layers: int, blocks: int=3, *args, **kwargs):
        super(ResVAE, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.device = device

        kernal_size=(3,3)
        channels = input_channels
        img_size = input_size

        encoder = []
        decoder = []

        encoder.append(nn.Conv2d(channels, 16 * channels, kernal_size, stride=1, padding=1))
        channels *= 16
        encoder.append(nn.ReLU())
        encoder.append(nn.BatchNorm2d(channels))

        for block in range(blocks):
            encoder.extend([ResBlock(channels, kernal_size)] * (num_layers // blocks))

            if block != blocks - 1:
                encoder.append(nn.Conv2d(channels, 2 * channels, 4, stride=2, padding=1))
                img_size = img_size // 2
                channels *= 2
                encoder.append(nn.ReLU())
                encoder.append(nn.BatchNorm2d(channels))

        encoder.append(nn.Flatten())
        self.encoder = nn.Sequential(*encoder)
        
        num_neurons = img_size ** 2 * channels

        self.mu= nn.Sequential(
            nn.Linear(num_neurons, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim)
        )

        self.logvar=nn.Sequential(
            nn.Linear(num_neurons, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim)
        )

        decoder.extend([
            nn.Linear(latent_dim, num_neurons),
            nn.ReLU(),
            nn.BatchNorm1d(num_neurons),
            nn.Unflatten(1, (channels, img_size, img_size))
        ])

        for block in range(blocks):
            if block != 0:
                decoder.append(nn.ConvTranspose2d(channels, channels // 2, 4, stride=2, padding=1))
                channels //= 2
                img_size *= 2
                decoder.append(nn.ReLU())
                decoder.append(nn.BatchNorm2d(channels))

            decoder.extend([ResBlock(channels, kernal_size)] * (num_layers // blocks))

        decoder.append(nn.Conv2d(channels, channels // 16, kernal_size, stride=1, padding=1))
        channels //= 16

        assert channels == input_channels, "number of output channels match the number of input channels"
        assert img_size == input_size, "size of output image does not match the size of the input image"

        decoder.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder)

class LinearVAE(VAE):
    """
    Shallow and wide linear VAE
    """
    def __init__(self, device, input_size: int, input_channels: int, latent_dim: int, *args, **kwargs):
        super(LinearVAE, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.device = device

        input_dim = input_size ** 2 * input_channels
        dim1 = 512
        dim2 = 256
        
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, dim1),
            nn.ReLU(),
            nn.Linear(dim1, dim2),
            nn.ReLU()
        )

        self.mu = nn.Linear(dim2, latent_dim)
        self.logvar = nn.Linear(dim2, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, dim2),
            nn.ReLU(),
            nn.Linear(dim2, dim1),
            nn.ReLU(),
            nn.Linear(dim1, input_dim),
            nn.Sigmoid(),
            nn.Unflatten(1, (input_channels, input_size, input_size))
        )

class ShallowConvVAE(VAE):
    """
    Shallow and wide conv VAE
    """
    def __init__(self, device, input_size: int, input_channels: int, latent_dim: int, *args, **kwargs):
        super(ShallowConvVAE, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.device = device

        self.latent_dim = latent_dim
        channels1 = 32
        channels2 = 64
        final_size = input_size // 2
        final_dim = channels2 * final_size ** 2
        pre_lat = 32
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, channels1, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels1, channels2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels2, channels2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels2, channels2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(final_dim, pre_lat),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, final_dim),
            nn.ReLU(),
            nn.Unflatten(1, (channels2, final_size, final_size)),
            nn.ConvTranspose2d(channels2, channels1, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels1, input_channels, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.mu = nn.Sequential(
            nn.Linear(pre_lat, self.latent_dim),
        )

        self.logvar = nn.Sequential(    
            nn.Linear(pre_lat, self.latent_dim),
        )
