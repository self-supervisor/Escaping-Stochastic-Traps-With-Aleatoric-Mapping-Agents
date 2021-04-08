import torch
import torch.nn.functional as F
import torch.nn.utils as nn
from .conversion_utils import scale_for_autoencoder


class ICM:
    def __init__(
        self,
        autoencoder,
        autoencoder_opt,
        uncertainty,
        device,
        preprocess_obss,
        uncertainty_budget=1,  # 0.1,  # 0.05,
        grad_clip=40,
    ):
        self.autoencoder = autoencoder
        self.autoencoder_opt = autoencoder_opt
        self.uncertainty = uncertainty
        self.grad_clip = grad_clip
        self.uncertainty_budget = uncertainty_budget
        self.preprocess_obss = preprocess_obss
        self.device = device
        self.predicted_frames = []
        self.predicted_uncertainty_frames = []
        self.count = 0

    def update_curiosity_parameters(self, loss):
        self.autoencoder_opt.zero_grad()
        loss.backward()
        self.autoencoder_opt.step()

    def compute_intrinsic_rewards(self, old_obs, new_obs, action):
        """Computes intrinsic rewards.

        Computes intrinsic rewards in parallel for different
        parallel agents. Also factors in aleatoric uncertainty
        quantification if desired.

        Returns
        -------
        loss : Torch Float
            Scalar loss of forward prediction model. Averaged
            over different parallel environments.
        reward: Torch Float Tensor
            (Parallel) Scalar intrinsic rewards as
            computed by forward pred.
        uncertainty: Torch Float
            Average uncertainty for loggin purposes.
        """
        new_obs = scale_for_autoencoder(
            self.preprocess_obss(new_obs, device=self.device).image, normalise=True
        )
        old_obs = scale_for_autoencoder(
            self.preprocess_obss(old_obs, device=self.device).image, normalise=True
        )
        self.count += 1
        # torch.save(action, f"action_{self.count}.pt")
        # torch.save(new_obs, f"new_obs_{self.count}.pt")
        # torch.save(old_obs, f"old_obs_{self.count}.pt")
        if self.uncertainty == "True":
            action_channel = torch.stack(
                [(torch.ones((7, 7, 1)) * an_action) / 6 for an_action in action]
            ).to(self.device)
            mu, sigma = self.autoencoder(old_obs, action_channel)
            mse = F.mse_loss(mu, new_obs, reduction="none")
            loss = torch.mean((torch.exp(-sigma) * mse + sigma), dim=(1, 2, 3))
            reward = torch.mean(torch.abs(mse - torch.exp(sigma)), dim=(1, 2, 3))
        else:
            action_channel = torch.stack(
                [(torch.ones((7, 7, 1)) * an_action) / 6 for an_action in action]
            ).to(self.device)
            mu, sigma = self.autoencoder(old_obs, action_channel)
            mse = F.mse_loss(mu, new_obs, reduction="none")
            loss = torch.mean(mse, dim=(1, 2, 3))
            reward = torch.mean(mse, dim=(1, 2, 3))
        uncertainty = torch.mean(sigma, dim=(1, 2, 3))
        reward *= 10

        return loss, reward, uncertainty
