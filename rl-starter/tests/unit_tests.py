import sys
import re
sys.path.append("..")  # quick and dirty works for now
import numpy as np
import pytest
import torch
from src.model import ACModel
from src.scripts.a2c import A2CAlgo
from src.scripts.models import AutoencoderWithUncertainty
import torch_ac

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
    frames_per_proc = None
    discount = 0.99
    lr = 0.001
    gae_lambda = 0.95
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    recurrence = 1
    optim_alpha = 0.99
    optim_eps = 1e-8
    envs = []
    procs = 16
    seed = 2
    env = "MiniGrid-KeyCorridorS6R3-v0"
    for i in range(procs):
        envs.append(make_env(env, seed + 10000 * i))

    obs_space, preprocess_obss = get_obss_preprocessor(envs[0].observation_space)

    acmodel = ACModel(obs_space, envs[0].action_space)

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
def test_update_visitation_counts(a2c_algo):
    for i in range(a2c_algo.visitation_counts.shape[0]):
        for j in range(a2c_algo.visitation_counts.shape[1]):
            assert np.count_nonzero(a2c_algo.visitation_counts[i][j]) == 0
    exps, logs1 = a2c_algo.collect_experiences()
    assert np.sum(a2c_algo.visitation_counts) == a2c_algo.num_frames_per_proc * a2c_algo.num_procs 
    exps, logs1 = a2c_algo.collect_experiences()
    assert np.sum(a2c_algo.visitation_counts) == a2c_algo.num_frames_per_proc * a2c_algo.num_procs * 2

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


import gym
import gym_minigrid


def make_env(env_key, seed=None):
    env = gym.make(env_key)
    env.seed(seed)
    return env


import gym
import gym_minigrid

def preprocess_images(images, device=None):                                                                                                       
    # Bug of Pytorch: very slow if not first converted to numpy array                                                                             
    images = np.array(images)                                                                                                                  
    return torch.tensor(images, device=device, dtype=torch.float)

def preprocess_texts(texts, vocab, device=None):
    var_indexed_texts = []
    max_text_len = 0

    for text in texts:
        tokens = re.findall("([a-z]+)", text.lower())
        var_indexed_text = np.array([vocab[token] for token in tokens])
        var_indexed_texts.append(var_indexed_text)
        max_text_len = max(len(var_indexed_text), max_text_len)

    indexed_texts = np.zeros((len(texts), max_text_len))

    for i, indexed_text in enumerate(var_indexed_texts):
        indexed_texts[i, :len(indexed_text)] = indexed_text

    return torch.tensor(indexed_texts, device=device, dtype=torch.long)

class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.vocab = {}

    def load_vocab(self, vocab):
        self.vocab = vocab

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]
