import matplotlib.pyplot as plt
import numpy as np
import os
import time


class time_ticker(object):
    def __init__(self):
        self.init_time = time.time()

    def begin(self):
        self.init_time = time.time()

    def tick(self, token, save_path=None):
        interval_time = time.time() - self.init_time
        self.init_time = time.time()
        # print('{0}: {1}'.format(token, interval_time))
        if save_path is not None:
            write_file(save_path, '{0}: {1}\n'.format(token, interval_time), False)
        return interval_time


def exist_or_create_folder(path_name):
    """
    Check whether a path exists, if not, then create this path.
    :param path_name: i.e., './logs/log.txt' or './logs/'
    :return: flag == False: failed; flag == True: successful.
    """
    flag = False
    pure_path = os.path.dirname(path_name)
    if not os.path.exists(pure_path):
        try:
            os.makedirs(pure_path)
            flag = True
        except OSError:
            pass
    return flag


def write_file(path, content, overwrite=False):
    """
    Write data to file.
    :param path:
    :param content:
    :param overwrite: open file by 'w' (True) or 'a' (False)
    :return:
    """
    exist_or_create_folder(path)
    if overwrite is True:
        with open(path, 'w') as f:
            f.write(str(content))
    else:
        with open(path, 'a') as f:
            f.write(content)


def read_file(path):
    """
    Read data from file.
    :param path:
    :return:
    """
    # Check the file path.
    if os.path.exists(os.path.dirname(path)):
        with open(path, 'r') as fo:
            data = fo.read()
    else:
        data = 'NONE'
    return data


def plot_rewards(rewards, savepath=None):
    # plt.plot(rewards, label='A2C')
    plt.plot(rewards)
    plt.title('running rewards in CartPole')  # plot figure title
    plt.xlabel('episodes')  # plot figure's x axis name.
    plt.ylabel('running rewards')  # plot figure's y axis name.
    y_axis_ticks = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]  # range of y axis
    plt.yticks(y_axis_ticks)  # set y axis's ticks
    for items in y_axis_ticks:  # plot some lines that vertical to y axis.
        plt.hlines(items, 0, len(rewards), colors="#D3D3D3", linestyles="dashed")
    # plt.legend(loc='best')
    if savepath is not None:
        exist_or_create_folder(savepath)
        plt.savefig(savepath + 'data.png')  # save figures.
    # plt.show()  # plt.show() must before plt.close()
    plt.close()


def read_plot_rewards(path, save_path):
    running_rewards = read_file(path)
    running_rewards = running_rewards.strip('[')
    running_rewards = running_rewards.strip(']')
    running_rewards = running_rewards.split(',')
    # rewards_list =
    rewards = []
    for item in running_rewards:
        rewards.append(float(item))
    plot_rewards(rewards, save_path)


def main():
    read_plot_rewards('./logs/999/rewards.txt', './logs/999/')


if __name__ == '__main__':
    main()
