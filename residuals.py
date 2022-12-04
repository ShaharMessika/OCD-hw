import torch.nn as nn
from copy import deepcopy

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class ConvMaxSpread(nn.Module):
    def __init__(self,dim = 256 ):
        super(ConvMaxSpread, self).__init__()

        self.Sequen = nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=2, stride=2),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, 5, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(8)])

        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.Flatten = nn.Flatten()
        self.Linear = nn.Linear(dim, 10)

    def forward(self, input):
        latent = []

        output = self.Sequen(input)

        output = self.AdaptiveAvgPool(output)
        output = self.Flatten(output)
        output = output.view(-1, self.num_flat_features(output))
        latent_in  = deepcopy(output.detach())
        output = self.Linear(output)
        latent = deepcopy(output.detach())


        return output,(latent,latent_in)


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

