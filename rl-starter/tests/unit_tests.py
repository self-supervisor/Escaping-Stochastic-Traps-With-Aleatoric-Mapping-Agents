import sys

sys.path.append("..")  # quick and dirty works for now
import pytest
from src.scripts.a2c import A2CAlgo
from src.scripts.models import AutoencoderWithUncertainty
from src.model import ACModel


@pytest.fixture
def a2c_algo():
    device = "cpu"
    icm_lr = 0.001
    autoencoder = AutoencoderWithUncertainty(observation_shape=(7, 7, 3)).to(device)
    autoencoder_opt = torch.optim.Adam(
        autoencoder.parameters(), lr=icm_lr, weight_decay=0
    )
    uncertainty = True
    noisy_tv = True
    curiosity = True
    randomise_env = False
    uncertainty_budget = 0.0005
    environment_seed = 1
    reward_weighting = 0.1
    normalise_rewards = True
    frames_before_reset = 2000
    frames_per_proc = 16
    discount = 0.99
    lr = 0.001
    gae_lambda = 0.95
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    recurrence = 1
    optim_alpha = 0.99
    optim_eps = 1e-8
    preprocess_obss = utils.get_obs_preprocessor(envs[0].observation_space)
    algo = A2CAlgo(
        envs,
        acmodel,
        autoencoder,
        autoencoder_opt,
        uncertainty,
        noisy_tv,
        curiosity,
        randomise_env,
        uncertainty_budget,
        environment_seed,
        reward_weighting,
        normalise_rewards,
        frames_before_reset,
        device,
        frames_per_proc,
        discount,
        lr,
        gae_lambda,
        entropy_coef,
        value_loss_coef,
        max_grad_norm,
        recurrence,
        optim_alpha,
        optim_eps,
        preprocess_obss,
    )
    return algo


# @pytest.mark.parametrize()
def test_update_visitation_counts():
    pass


# @pytest.mark.parametrize()
def test_add_noisy_tv():
    pass


# @pytest.mark.parametrize()
def test_reset_environments_if_ness():
    pass


# @pytest.mark.parametrize()
def test_compute_intrinsic_rewards():
    pass


def test_get_mean_and_std_dev():
    pass


def test_get_label_from_path():
    pass


def test_plot():
    pass


def get_obss_preprocessor(obs_space):
    """
    relative import hack
    """
    # Check if obs_space is an image space
    if isinstance(obs_space, gym.spaces.Box):
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({"image": preprocess_images(obss, device=device)})

    # Check if it is a MiniGrid observation space
    elif isinstance(obs_space, gym.spaces.Dict) and list(obs_space.spaces.keys()) == [
        "image"
    ]:
        obs_space = {"image": obs_space.spaces["image"].shape, "text": 100}

        vocab = Vocabulary(obs_space["text"])

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList(
                {
                    "image": preprocess_images(
                        [obs["image"] for obs in obss], device=device
                    ),
                    "text": preprocess_texts(
                        [obs["mission"] for obs in obss], vocab, device=device
                    ),
                }
            )

        preprocess_obss.vocab = vocab

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss
