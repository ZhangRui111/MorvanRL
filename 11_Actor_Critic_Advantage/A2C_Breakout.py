"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.
The Breakout example. Policy is oscillated.
"""

import cv2
import gym
import numpy as np
import os
import tensorflow as tf
import time

from utils import write_file, plot_rewards, restore_parameters, save_parameters

# np.random.seed(2)
# tf.set_random_seed(2)  # reproducible
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


OUTPUT_GRAPH = False
RENDER = False  # rendering wastes time
CROP_SIZE = 80
N_A = 4
DISPLAY_REWARD_THRESHOLD = 100  # renders environment if total episode reward is greater then this threshold
SAVED_INTERVAL = 1000
y_axis_ticks = [-10, -5, 0]
weights_path = './logs/Breakout/weights/'
data_path = './logs/Breakout/data/'


MAX_EPISODE = 10001
GAMMA = 0.9  # reward discount in TD error: 0.9
LR_A = 0.0001  # learning rate for actor: 0.0001
LR_C = 0.001  # learning rate for critic: 0.001


def preprocess_image(img):
    img = img[30:-15, 5:-5:, :]  # image cropping
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert from BGR to GRAY
    gray = cv2.resize(gray, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_NEAREST)
    return gray


class Actor(object):
    def __init__(self, sess, crop_size, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, crop_size, crop_size], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            input_crop = self.s / 255
            input = tf.transpose(input_crop, [1, 2, 0])
            conv1 = tf.contrib.layers.conv2d(inputs=input[np.newaxis, :], num_outputs=32, kernel_size=8, stride=4)
            conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=32, kernel_size=4, stride=2)
            conv3 = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=32, kernel_size=3, stride=1)

            flat = tf.contrib.layers.flatten(conv3)
            f = tf.contrib.layers.fully_connected(flat, 512)
            self.acts_prob = tf.contrib.layers.fully_connected(f, n_actions, activation_fn=tf.nn.softmax)

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})  # get probabilities for all actions
        action = np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())
        return action, probs.flatten()


class Critic(object):
    def __init__(self, sess, crop_size, lr=0.01):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, crop_size, crop_size], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            input_crop = self.s / 255
            input = tf.transpose(input_crop, [1, 2, 0])
            conv1 = tf.contrib.layers.conv2d(inputs=input[np.newaxis, :], num_outputs=32, kernel_size=8, stride=4)
            conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=32, kernel_size=4, stride=2)
            conv3 = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=32, kernel_size=3, stride=1)

            flat = tf.contrib.layers.flatten(conv3)
            f = tf.contrib.layers.fully_connected(flat, 512)
            self.v = tf.contrib.layers.fully_connected(f, 1, activation_fn=None)

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    feed_dict={self.s: s, self.v_: v_, self.r: r})
        return td_error


def main():
    env = gym.make('Breakout-v0')
    # env.seed(1)  # reproducible
    # env = env.unwrapped
    gpu_options = tf.GPUOptions(allow_growth=True)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    actor = Actor(sess, crop_size=CROP_SIZE, n_actions=N_A, lr=LR_A)
    critic = Critic(sess, crop_size=CROP_SIZE, lr=LR_C)

    sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/Breakout/", sess.graph)

    episodes = []
    episode_rewards = []
    running_rewards = []
    total_steps = 0
    running_reward = -10

    saver, load_episode = restore_parameters(sess, weights_path)
    write_file(data_path + 'probs.txt', 'probs\n', True)

    for i_episode in range(MAX_EPISODE):
        s = env.reset()
        s = preprocess_image(s)

        episode_steps = 0
        track_r = []
        while True:
            # if RENDER:
            #     env.render()
            a, probs = actor.choose_action(s)
            print(probs)
            probs = np.around(probs, decimals=4)
            write_file(data_path + 'probs.txt', str([i_episode, total_steps]) + '  ' + str(probs.tolist()) + '\n', False)
            # if episode_steps % 20 == 0:  # episode_steps % 10 --> reserve the ball.
            #     a = 1

            # a = np.random.random_integers(0, 3)

            s_, r, done, info = env.step(a)
            s_ = preprocess_image(s_)

            if done:
                r = -10  # -20

            track_r.append(r)

            td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(s, a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]

            s = s_
            episode_steps += 1
            total_steps += 1

            if done:
                ep_rs_sum = sum(track_r)

                episodes.append(episodes)
                episode_rewards.append(ep_rs_sum)

                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05

                running_rewards.append(running_reward)

                if len(running_rewards) % SAVED_INTERVAL == 0:
                    write_file(data_path + 'rewards_' + str(i_episode) + '.txt', running_rewards, True)
                    plot_rewards(running_rewards, y_axis_ticks, data_path)
                if i_episode % SAVED_INTERVAL == 0 and i_episode != 0:
                    save_parameters(sess, weights_path, saver,
                                    weights_path + '-' + str(load_episode + i_episode))

                # if running_reward > DISPLAY_REWARD_THRESHOLD:
                #     RENDER = True  # rendering
                print("---------------------------------------------------------------episode:", i_episode,
                      " episode  reward:", ep_rs_sum, " running  reward:", round(running_reward, 4))
                # time.sleep(0.5)
                break


if __name__ == '__main__':
    main()
