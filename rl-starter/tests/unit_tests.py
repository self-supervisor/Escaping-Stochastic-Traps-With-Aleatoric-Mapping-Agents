import sys

sys.path.append("..")  # quick and dirty works for now
import re
import numpy as np
import pytest
import torch
from src.model import ACModel
from src.scripts.a2c import A2CAlgo
from src.scripts.models import AutoencoderWithUncertainty
import torch_ac

@pytest.fixture(scope="module", params=[[True, True, True], 
                                        [True, False, True],
                                        [False, True, True],
                                        [False, False, True],
                                        [False, True, False],
                                        [False, False, False]])
def a2c_algo(request):
    device = "cpu"
    icm_lr = 0.001
    autoencoder = AutoencoderWithUncertainty(observation_shape=(7, 7, 3)).to(device)
    autoencoder_opt = torch.optim.Adam(
        autoencoder.parameters(), lr=icm_lr, weight_decay=0
    )
    uncertainty = request.param[0]
    noisy_tv = request.param[1]
    curiosity = request.param[2]
    randomise_env = "False"
    uncertainty_budget = 0.0005
    environment_seed = 1
    reward_weighting = 0.1
    normalise_rewards = True
    frames_before_reset = 8
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


def test_update_visitation_counts(a2c_algo):
    for i in range(a2c_algo.visitation_counts.shape[0]):
        for j in range(a2c_algo.visitation_counts.shape[1]):
            assert np.count_nonzero(a2c_algo.visitation_counts[i][j]) == 0
    exps, logs1 = a2c_algo.collect_experiences()
    assert np.sum(a2c_algo.visitation_counts) == a2c_algo.num_frames_per_proc * a2c_algo.num_procs 
    exps, logs1 = a2c_algo.collect_experiences()
    assert np.sum(a2c_algo.visitation_counts) == a2c_algo.num_frames_per_proc * a2c_algo.num_procs * 2


def test_add_noisy_tv(a2c_algo):
    action = [1] * 15 + [6]
    obs, _, _, _ = a2c_algo.env.step(action)
    copy_changed = obs[15]["image"].copy()
    copy_same = obs[0]["image"].copy()
    obs_with_tv = a2c_algo.add_noisy_tv(obs, action, a2c_algo.env.envs)
    assert np.array_equal(copy_same, obs_with_tv[0]["image"])
    assert np.array_equal(copy_changed, obs_with_tv[-1]["image"]) == False
    

def test_reset_environments_if_ness(a2c_algo):
    failure_count = 0
    for _ in range(10):
        a2c_algo.frames_before_reset = 8 
        _, _ = a2c_algo.collect_experiences()

        a2c_algo.reset_environments_if_ness(0)
        grids = []
        positions = []
        for an_env in a2c_algo.env.envs:
            positions.append(an_env.agent_pos)
            grids.append(str(an_env))

        assert all_same(grids)
        assert all_same(positions)

        a2c_algo.frames_before_reset = 9 
        _, _ = a2c_algo.collect_experiences()
        a2c_algo.reset_environments_if_ness(0)
        
        grids = []
        positions = []
        for an_env in a2c_algo.env.envs:
            positions.append(an_env.agent_pos)
            grids.append(str(an_env))
        # hack because sometimes this is true sometimes 
        # policy is same across envs after all 
        try:
            assert all_same(grids) == False
            assert all_same(positions) == False
        except:
            failure_count += 1
        if failure_count == 10:
            assert True == False

def test_get_mean_and_std_dev():
    from src.scripts.plot import get_mean_and_std_dev 

    make_fake_csvs(zeros=True)
    csv_paths = ["fake1.csv", "fake2.csv", "fake3.csv", "fake4.csv"]
    mean, std_dev = get_mean_and_std_dev(csv_paths, "quantity_one") 
    mean, std_dev = get_mean_and_std_dev(csv_paths, "quantity_two") 

    for index, a_val in enumerate(mean):
        assert a_val == 0
        assert std_dev[index] == 0
   
    clean_csvs()

    make_fake_csvs(zeros=False)
    csv_paths = ["fake1.csv", "fake2.csv", "fake3.csv", "fake4.csv"]
    mean, std_dev = get_mean_and_std_dev(csv_paths, "quantity_one") 
    mean, std_dev = get_mean_and_std_dev(csv_paths, "quantity_two") 

    for index, a_val in enumerate(mean):
        assert std_dev[index] != 0

    clean_csvs()

def test_get_label_from_path():
    from src.scripts.plot import get_label_from_path 

    example_paths = ["storage/frames_8_noisy_tv_True_curiosity_False_uncertainty_False_random_seed_86_coefficient_0.0005",
                     "storage/frames_8_noisy_tv_False_curiosity_True_uncertainty_True_random_seed_85_coefficient_0.0005_MiniGrid-KeyCorridorS6R3-v0", 
                     "storage/frames_8_noisy_tv_True_curiosity_True_uncertainty_False_random_seed_89_coefficient_0.0005"]
    labels = [get_label_from_path(a_path) for a_path in example_paths] 
    assert labels[0] == "frames 8 noisy tv True curiosity False uncertainty False"
    assert labels[1] == "frames 8 noisy tv False curiosity True uncertainty True"
    assert labels[2] == "frames 8 noisy tv True curiosity True uncertainty False"

def test_plot():
    import glob
    import os
    import shutil

    make_fake_csvs(zeros=True, for_plot=True)
    from src.scripts.plot import plot
    path_strings = glob.glob("storage/*/*csv")
    path_strings = [a_string[:-8] for a_string in path_strings]
    plot("software testing plot", [path_strings], "quantity_one") 
    plot("software testing plot", [path_strings], "quantity_two") 
    shutil.rmtree("storage")

    make_fake_csvs(zeros=False, for_plot=True)
    from src.scripts.plot import plot
    path_strings = glob.glob("storage/*/*csv")
    path_strings = [a_string[:-8] for a_string in path_strings]
    plot("software testing plot no zeros", [path_strings], "quantity_one") 
    plot("software testing plot no zeros", [path_strings], "quantity_two") 
    shutil.rmtree("storage")

def make_fake_csvs(zeros, for_plot=False):
    import csv 
    import random
    import os

    for run in range(4):
        run_number = run + 1
        if for_plot:
            os.makedirs(f"storage/frames_8_noisy_tv_True_curiosity_True_uncertainty_False_random_seed_89_coefficient_0.0005_fake{run_number}")
            with open(f"storage/frames_8_noisy_tv_True_curiosity_True_uncertainty_False_random_seed_89_coefficient_0.0005_fake{run_number}/log.csv", "w", newline="\n") as csvfile:
                writer = csv.writer(csvfile, delimiter=",")
                for row_index in range(1000):
                    writer.writerow(["update", "quantity_one", "quantity_two"])
                    if zeros:
                        writer.writerow([0, 0, 0])
                    else:
                        writer.writerow([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0,1)])
        else:
            with open(f"fake{run_number}.csv", "w", newline="\n") as csvfile:
                writer = csv.writer(csvfile, delimiter=",")
                for row_index in range(1000):
                    writer.writerow(["update", "quantity_one", "quantity_two"])
                    if zeros:
                        writer.writerow([0, 0, 0])
                    else:
                        writer.writerow([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0,1)])


def clean_csvs():
    import glob
    import os 

    files=glob.glob("fake*.csv")
    for filename in files:
        os.unlink(filename)

def all_same(items):
    """
    https://stackoverflow.com/questions/3787908/
    python-determine-if-all-items-of-a-list-
    are-the-same-item
    """
    return all(np.array_equal(x, items[0]) for x in items)

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