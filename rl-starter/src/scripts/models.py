import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


# define the NN architecture
class AutoencoderWithUncertainty(nn.Module):
    def __init__(self):
        super(AutoencoderWithUncertainty, self).__init__()
        self.image_conv = nn.Sequential(
            nn.Conv2d(4, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
        )

        self.image_deconv_mu = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (3, 3)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (3, 3)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, (3, 3)),
        )
        self.image_deconv_sigma = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 147),
        )
        torch.nn.init.constant(self.image_deconv_sigma[0].weight, -2)
        torch.nn.init.constant(self.image_deconv_sigma[2].weight, -2)
        torch.nn.init.constant(self.image_deconv_sigma[4].weight, -2)

    def forward(self, x, action_channel):
        an_input = torch.cat([x, action_channel], dim=3)
        an_input = an_input.reshape((128, 4, 7, 7))
        z = self.image_conv(an_input)
        mu = self.image_deconv_mu(z)
        sigma = self.image_deconv_sigma(z.reshape((128, 64)))
        mu = mu.reshape((128, 7, 7, 3))
        sigma = sigma.reshape((128, 7, 7, 3))
        return mu, sigma

        forward_prediction = self.forward_predictor(embedding).view(-1, 7, 7, 3)
        uncertainty = self.uncertainty_predictor(embedding).view(-1, 7, 7, 3)
        return forward_prediction, uncertainty
