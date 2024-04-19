"""
Plotting code for comparative analysis of all methods.
"""

import numpy as np
import matplotlib.pyplot as plt


def moving_average(data, *, window_size=50):
    """Smooths 1-D data array using a moving average.

    Args:
        data: 1-D numpy.array
        window_size: Size of the smoothing window

    Returns:
        smooth_data: A 1-d numpy.array with the same size as data
    """
    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(
        np.ones_like(data), kernel
    )
    return smooth_data[: -window_size + 1]


def main():

    returns_dqn = np.load('/home/kaushal/spring_2024/CS5180_RL/learning-on-the-highway/q_learning_based_methods/returns_highway_dqn.npy')  # noqa
    returns_dqn_per = np.load('/home/kaushal/spring_2024/CS5180_RL/learning-on-the-highway/q_learning_based_methods/returns_hwy_dqn_per.npy')  # noqa
    returns_ddqn_per = np.load('/home/kaushal/spring_2024/CS5180_RL/learning-on-the-highway/q_learning_based_methods/returns_hwy_ddqn_per.npy')  # noqa

    # # Lighter Red color
    # light_red = (1.0, 0.6, 0.6)
    # # Lighter Green color
    # light_green = (0.6, 1.0, 0.6)
    # # Ligher Blue color
    # light_blue = (0.6, 0.6, 1.0)

    # plt.plot(returns_dqn, label='Raw Data DQN', color=light_red)
    data = moving_average(data=returns_dqn, window_size=1000)
    plt.plot(data, label='Smoothened Data DQN', color='red')
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Returns')

    # plt.plot(returns_dqn_per, label='Raw Data', color=light_green)
    data = moving_average(data=returns_dqn_per, window_size=1000)
    plt.plot(data, label='Smoothened data DQN with PER', color='green')
    plt.legend()
    plt.title('Returns For Highway Fast: DQN, Prioritized Experience Replay')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')

    # plt.plot(returns_ddqn_per, label='Raw Data', color=light_blue)
    data = moving_average(data=returns_ddqn_per, window_size=1000)
    plt.plot(data, label='Smoothened data DDQN with PER', color='blue')
    plt.legend()
    plt.title('Returns For Highway Fast: DDQN, Prioritized Experience Replay')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
