import gym
from baselines.common.atari_wrappers import NoopResetEnv, FrameStack
from wrappers import (
    MontezumaInfoWrapper,
    make_mario_env,
    make_robo_pong,
    make_robo_hockey,
    make_multi_pong,
    AddRandomStateToInfo,
    MaxAndSkipEnv,
    ProcessFrame84,
    ExtraTimeLimit,
    StickyActionEnv,
    NoisyTVEnvWrapper,
    ActionLoggingWrapper,
)
import numpy as np
import time


class BankHeistVideoRecorder:
    def __init__(self, policy, args):
        self.policy = policy
        self.env = self.make_atari_env(args)
        self.car_mask = {
            "104, 72, 98": np.zeros((250, 160)),
            "66, 158, 130": np.zeros((250, 160)),
            "252, 252, 84": np.zeros((250, 160)),
            "236, 200, 96": np.zeros((250, 160)),
        }

    def update_car_mask(self, car_color=np.array([223, 183, 85])):
        for a_frame in self.frames:
            maze_count = 0
            for a_key in self.car_mask.keys():
                if np.array([int(i) for i in a_key.split(",")]) in a_frame:
                    maze_count += 1
                    self.car_mask[a_key] += np.sum(
                        np.where(np.array(a_frame) == car_color, [1, 1, 1], [0, 0, 0]),
                        axis=2,
                    )
            assert maze_count < 2  # should only be in one maze at a time

    def record_episode(self):
        self.frames = []
        obs = self.env.reset()
        obs = self.prepare_obs_for_pol(obs)
        action, _, _ = self.policy.get_ac_value_nlp(obs)
        for _ in range(4500):
            obs, reward, done, info = self.env.step(action[0])
            obs = self.prepare_obs_for_pol(obs)
            self.frames.append(self.env.unwrapped.render("rgb_array"))
            action, _, _ = self.policy.get_ac_value_nlp(obs)

        self.update_car_mask()
        self.plot_car_mask()

    def plot_car_mask(self):
        import wandb
        import matplotlib.pyplot as plt
        import time

        unix_timestamp = time.time()
        image = np.zeros((250, 160 * 4))
        for i, a_key in enumerate(self.car_mask.keys()):
            image[:, i * 160 : (i + 1) * 160] = self.car_mask[a_key]

        plt.imshow(np.log(image))
        plt.colorbar()
        wandb.log({f"car_mask_plot_log_scale_{unix_timestamp}": plt})
        wandb.log({"novel states": np.sum(np.where(np.log(image) == -np.inf, 0, 1))})
        plt.close()
        np.save(f"car_mask_{unix_timestamp}.npy", image)
        np.save(f"frames_{unix_timestamp}.npy", self.frames[0:100])

    def prepare_obs_for_pol(self, obs):
        obs = np.array(obs)
        obs = np.squeeze(obs, axis=3)
        obs = np.reshape(obs, (1, 84, 84, 4))
        return obs

    def make_atari_env(self, args):
        """
        duplicated code hack due to
        relative import errors
        """

        env = gym.make(args["env"])
        assert "NoFrameskip" in env.spec.id
        # from self-supervised exploration via disagreement
        if args["stickyAtari"] == "true":
            env = StickyActionEnv(env)
        env._max_episode_steps = args["max_episode_steps"] * 4
        env = MaxAndSkipEnv(env, skip=4)
        env = ProcessFrame84(env, crop=False)
        env = FrameStackNoLazy(env, 4)
        env = ExtraTimeLimit(env, args["max_episode_steps"])
        if "Montezuma" in args["env"]:
            env = MontezumaInfoWrapper(env)
        env = AddRandomStateToInfo(env)
        if args["noisy_tv"] == "true":
            env = NoisyTVEnvWrapper(env)
        return env


class FrameStackNoLazy(FrameStack):
    def _get_ob(self):
        assert len(self.frames) == self.k
        return list(self.frames)
