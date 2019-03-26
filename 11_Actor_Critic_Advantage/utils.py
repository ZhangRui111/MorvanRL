import cv2
# --------Remove these lines if plot.show() in PC----------------#
import matplotlib as mlp
mlp.use('Agg')
# --------Remove these lines if plot.show() in PC----------------#
import matplotlib.pyplot as plt
import os
import time
import tensorflow as tf


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


def preprocess_image(img, crop_size):
    # print('0')
    # plt.imshow(img)
    # plt.show()
    # plt.close()
    img = img[30:-15, 5:-5:, :]  # image cropping
    # print('1')
    # plt.imshow(img)
    # plt.show()
    # plt.close()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert from BGR to GRAY
    # print('2')
    # plt.imshow(img, cmap="gray")
    # plt.show()
    # plt.close()
    gray = cv2.resize(gray, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
    # print('3')
    # plt.imshow(img, cmap="gray")
    # plt.show()
    # plt.close()
    return gray


def show_gray_image(img):
    plt.imshow(img, cmap="gray")
    plt.show()
    plt.close()


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
            f.write(str(content))


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


def plot_rewards(rewards, y_axis_ticks, savepath=None):
    # plt.plot(rewards, label='A2C')
    plt.plot(rewards)
    plt.title('running rewards in CartPole')  # plot figure title
    plt.xlabel('episodes')  # plot figure's x axis name.
    plt.ylabel('running rewards')  # plot figure's y axis name.
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


def restore_parameters(sess, restore_path):
    saver = tf.train.Saver(max_to_keep=10)
    checkpoint = tf.train.get_checkpoint_state(restore_path)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
        path_ = checkpoint.model_checkpoint_path
        step = int((path_.split('-'))[-1])
    else:
        # Re-train the network from zero.
        print("Could not find old network weights")
        step = 0
    return saver, step


def save_parameters(sess, save_path, saver, name):
    exist_or_create_folder(save_path)
    saver.save(sess, name)


def main():
    read_plot_rewards('./logs/999/rewards.txt', './logs/999/')


if __name__ == '__main__':
    main()
