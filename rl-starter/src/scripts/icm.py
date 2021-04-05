import torch
import torch.nn.functional as F
import torch.nn.utils as nn
from .conversion_utils import scale_for_forward_model


class ICM:
    def __init__(
        self,
        forward_model,
        forward_model_opt,
        uncertainty,
        device,
        preprocess_obss,
        reward_weighting,
        uncertainty_budget=1,  # 0.005,
        grad_clip=40,
    ):
        self.forward_model = forward_model
        self.forward_model_opt = forward_model_opt
        self.uncertainty = uncertainty
        self.grad_clip = grad_clip
        self.uncertainty_budget = uncertainty_budget
        self.preprocess_obss = preprocess_obss
        self.device = device
        self.predicted_frames = []
        self.predicted_uncertainty_frames = []
        self.reward_weighting = reward_weighting

    def update_curiosity_parameters(self, loss):
        self.forward_model_opt.zero_grad()
        loss.backward()
        # nn.clip_grad_norm_(self.forward_model.parameters(), self.grad_clip)
        self.forward_model_opt.step()

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
        new_obs = scale_for_forward_model(
            self.preprocess_obss(new_obs, device=self.device).image, normalise=True
        )
        old_obs = scale_for_forward_model(
            self.preprocess_obss(old_obs, device=self.device).image, normalise=True
        )
        action_vector = torch.tensor(torch.stack([action] * 16), dtype=torch.float).to(
            self.device
        )
        action_vector /= 6
        forward_prediction, uncertainty = self.forward_model(old_obs, action_vector)
        self.predicted_frames.append(forward_prediction)
        self.predicted_uncertainty_frames.append(uncertainty)
        if self.uncertainty == "True":
            mse = F.mse_loss(forward_prediction, new_obs, reduction="none")
            loss = 0.5 * torch.sum(torch.exp(-uncertainty) * mse + uncertainty)
            reward = torch.clamp(
                (torch.mean(mse - torch.exp(uncertainty), dim=(1, 2, 3))), -5, 5
            )
        else:
            reward = F.mse_loss(forward_prediction, new_obs, reduction="none")
            reward = torch.mean(reward, dim=(1, 2, 3))
            loss = torch.sum(reward)
        reward *= self.reward_weighting
        uncertainty = torch.mean(uncertainty, dim=(1, 2, 3))
        return loss, reward, uncertainty
