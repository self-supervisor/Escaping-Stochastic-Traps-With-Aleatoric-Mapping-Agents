import gym
import numpy as np

class NoisyTVWrapper(gym.Wrapper):
    def __init__(self, env, noisy_tv):
        super().__init__(env)
        self.env = env
        self.noisy_tv = noisy_tv

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if self.noisy_tv == "True": 
            next_state = self.add_noisy_tv(next_state, action)
        return next_state, reward, done, info

    def add_noisy_tv(self, obs_tp1, action):
        """
        Noisy TV dependent on the done (6) action.
        The range of the noisy TV random ints comes
        from the number of valid values each channel
        can theoretically have in the gym minigrid
        environment.
        Returns
        -------
        obs_tp1 : numpy array
            Returns frames one step into the future that
            have the action dependent noisy TV injected.
        """
        import random

        for i, a_action in enumerate(action):
            if action[i] == 6:
                a = np.random.randint(0, 6, (5, 5, 1))
                b = np.random.randint(0, 11, (5, 5, 1))
                c = np.random.randint(0, 3, (5, 5, 1))
                obs_tp1[i]["image"][0:5, 0:5, :] = np.squeeze(
                    np.stack([a, b, c], axis=3)
                )
        return obs_tp1
