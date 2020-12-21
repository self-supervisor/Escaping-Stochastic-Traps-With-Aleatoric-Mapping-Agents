import numpy as np
import pytest


@pytest.fixture
def noisy_mnist_env():
    from noisy_mnist_aleatoric_uncertainty_for_poster import NoisyMnistEnv

    mnist_env = NoisyMnistEnv("train", 0, 2)
    return mnist_env


def check_count_of_classes(x_arr, y_arr):
    same = 0
    not_same = 0
    for i, _ in enumerate(x_arr):
        if np.array_equal(x_arr[i], y_arr[i]):
            same += 1
        else:
            not_same += 1
    return same, not_same


def test_mnist_env_step(noisy_mnist_env):
    import math

    x_arr, y_arr = noisy_mnist_env.step()
    assert x_arr.shape == y_arr.shape  # make sure batch shapes make sense
    for i, _ in enumerate(x_arr):  # check batch is completely filled
        assert np.array_equal(x_arr[i], np.zeros((1, 28 * 28))) == False
        assert np.array_equal(y_arr[i], np.zeros((1, 28 * 28))) == False
        assert np.array_equal(np.zeros((1, 28 * 28)), np.zeros((1, 28 * 28))) == True
    same = 0
    not_same = 0
    for _ in range(
        1000
    ):  # check roughly half are deterministic transitions, half aren't
        x_arr, y_arr = noisy_mnist_env.step()
        same_sample, not_same_sample = check_count_of_classes(x_arr, y_arr)
        same += same_sample
        not_same += not_same_sample
    print("same", same)
    print("not same", not_same)
    assert math.isclose(same, not_same, rel_tol=0.2)


def test_mnist_env_random_sample_of_number(noisy_mnist_env):
    """
    This test is a qualitative visual test, look in test images
    and make sure the number title is the same as the number
    """
    import matplotlib.pyplot as plt
    import os
    import shutil

    if os.path.isdir("test_images"):
        shutil.rmtree("test_images")
    os.mkdir("test_images")

    for number in range(0, 10):
        digit = noisy_mnist_env.get_random_sample_of_number(number)
        plt.imshow(np.array(digit).reshape(28, 28))
        plt.title(str(number))
        plt.savefig("test_images/" + str(number) + ".png")
