#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gym
import gym_minigrid
import pytest
import numpy as np


@pytest.fixture
def key_corridor_altered():
    env = gym.make("MiniGrid-KeyCorridorS6R3-v0")
    return env


def test_not_procedural(key_corridor_altered):
    import random

    _ = key_corridor_altered.reset()
    initial_position = key_corridor_altered.agent_pos
    for _ in range(1000):
        done = False
        while done == False:
            obs, reward, done, info = key_corridor_altered.step(random.randint(0, 6))
        key_corridor_altered.step(random.randint(0, 6))
        assert np.array_equal(key_corridor_altered.agent_pos, initial_position)


def test_episode_length():
    # loop through 8 steps
    # assert environment resets
    # and returns done after 8 steps
    pass
