import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()
        self.res = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel, 3, 1, bias=False),
            nn.InstanceNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel, 3, 1, bias=False),
            nn.InstanceNorm2d(in_channel)
        )
        self.relu = nn.ReLU()

    
    def forward(self, x):
        out = self.res(x)
        return x + out


class Generator(nn.Module):
    def __init__(self, in_channel=3) -> None:
        super().__init__()
        self.gen = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channel, 64, 7, 1, bias=False),

            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(),

            ResNet(256),
            ResNet(256),
            ResNet(256),
            ResNet(256),
            ResNet(256),
            ResNet(256),
            ResNet(256),
            ResNet(256),
            ResNet(256),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7, 1, bias=False),
            nn.Tanh()
        )


    def forward(self, x):
        out = self.gen(x)
        return out
    

class Discriminator(nn.Module):
    def __init__(self, in_channel=3) -> None:
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(in_channel, 64, 4, 2, 1, bias=False),
            # nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 1, bias=False),
            # nn.Sigmoid()
        )


    def forward(self, x):
        out = self.disc(x)
        return out





