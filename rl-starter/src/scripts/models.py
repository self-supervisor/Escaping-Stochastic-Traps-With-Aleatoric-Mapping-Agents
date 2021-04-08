import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

# define the NN architecture
class AutoencoderWithUncertainty(nn.Module):
    def __init__(self):
        super(AutoencoderWithUncertainty, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, (2, 2))
        self.conv2 = nn.Conv2d(16, 32, (2, 2))
        self.conv3 = nn.Conv2d(32, 64, (2, 2))
        self.mu_conv1 = nn.ConvTranspose2d(64, 32, (3, 3))
        self.mu_conv2 = nn.ConvTranspose2d(32, 16, (3, 3))
        self.mu_conv3 = nn.ConvTranspose2d(16, 3, (3, 3))
        self.sigma_1 = nn.Linear(64, 128)
        self.sigma_2 = nn.Linear(128, 128)
        self.sigma_3 = nn.Linear(128, 147)
        self.linear_mu_out = nn.Linear(147 * 2 + (7 * 7), 147)
        self.linear_sigma_out = nn.Linear(147 * 2 + (7 * 7), 147)

    #         torch.nn.init.constant(self.image_deconv_sigma[0].weight, -2)
    #         torch.nn.init.constant(self.image_deconv_sigma[2].weight, -2)
    #         torch.nn.init.constant(self.image_deconv_sigma[4].weight, -2)

    def forward(self, x, action_channel):
        batch_size, _, _, channels = x.shape
        an_input = torch.cat([x, action_channel], dim=3)
        an_input = an_input.reshape((batch_size, channels + 1, 7, 7))
        x1 = F.relu(self.conv1(an_input))
        x1 = F.max_pool2d(x1, (2, 2))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        mu1 = F.relu(self.mu_conv1(x3))
        mu2 = F.relu(self.mu_conv2(mu1))
        mu3 = self.mu_conv3(mu2)
        mu3 = torch.flatten(mu3, start_dim=1)
        an_input = torch.flatten(an_input, start_dim=1)
        mu = torch.cat((mu3, an_input), dim=1)
        mu = self.linear_mu_out(mu)
        x3 = x3.reshape(batch_size, -1)
        sigma1 = F.relu(self.sigma_1(x3))
        sigma2 = F.relu(self.sigma_2(sigma1))
        sigma = self.sigma_3(sigma2)
        sigma = torch.cat((sigma, an_input), dim=1)
        sigma = self.linear_sigma_out(sigma)
        sigma = sigma.reshape(batch_size, 7, 7, 3)
        mu = mu.reshape(batch_size, 7, 7, 3)
        return mu, sigma
