import torch
import torch.nn.functional as F
import torch.nn.utils as nn
from .conversion_utils import scale_for_autoencoder


class ICM:
    def __init__(
        self,
        autoencoder_ama,
        autoencoder_ama_opt,
        autoencoder_mse,
        autoencoder_mse_opt,
        uncertainty,
        device,
        preprocess_obss,
        uncertainty_budget=1,  # 0.1,  # 0.05,
        grad_clip=40,
    ):
        self.autoencoder_mse = autoencoder_mse
        self.autoencoder_mse_opt = autoencoder_mse_opt
        self.autoencoder_ama = autoencoder_ama
        self.autoencoder_ama_opt = autoencoder_ama_opt
        self.uncertainty = uncertainty
        self.grad_clip = grad_clip
        self.uncertainty_budget = uncertainty_budget
        self.preprocess_obss = preprocess_obss
        self.device = device
        self.predicted_frames = []
        self.predicted_uncertainty_frames = []
        self.count = 0

    def update_curiosity_parameters(self, loss_mse, loss_ama):
        loss = loss_mse + loss_ama
        self.autoencoder_mse_opt.zero_grad()
        self.autoencoder_ama_opt.zero_grad()
        loss.backward()
        self.autoencoder_mse_opt.step()
        self.autoencoder_ama_opt.step()

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
        action_vector = torch.tensor(torch.stack([action] * 109), dtype=torch.float).to(
            self.device
        )
        action_vector /= 6
        forward_prediction_mse, uncertainty_mse = self.autoencoder_mse(
            old_obs, action_vector
        )
        forward_prediction_ama, uncertainty_ama = self.autoencoder_ama(
            old_obs, action_vector
        )
        self.count += 1
        torch.save(action, f"action_{self.count}.pt")
        torch.save(old_obs, f"old_obs_{self.count}.pt")
        torch.save(new_obs, f"new_obs_{self.count}.pt")
        if self.uncertainty == "True":
            mse_ama = F.mse_loss(forward_prediction_ama, new_obs, reduction="none")
            mse_ama_copy = mse_ama.detach().clone()
            loss_ama = torch.sum(
                torch.mean(
                    (
                        torch.exp(-uncertainty_ama) * mse_ama
                        + self.uncertainty_budget * uncertainty_ama
                    ),
                    dim=(1, 2, 3),
                )
            )
            print("loss ama", torch.sum(loss_ama))
            mse_ama = torch.mean(mse_ama_copy, dim=(1, 2, 3))
            reward = F.mse_loss(forward_prediction_mse, new_obs, reduction="none")
            reward = torch.mean(reward, dim=(1, 2, 3))
            loss_mse = torch.sum(reward)
        else:
            reward = F.mse_loss(forward_prediction, new_obs, reduction="none")
            reward = torch.mean(reward, dim=(1, 2, 3))
            loss = torch.sum(reward)
            mse = reward
        uncertainty = torch.mean(uncertainty_ama, dim=(1, 2, 3))
        return loss_mse, loss_ama, mse_ama, reward, uncertainty
