"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.
The Pong example. Policy is oscillated.
"""
import gym
import numpy as np
import os
import tensorflow as tf

from utils import write_file, plot_rewards, preprocess_image, restore_parameters, save_parameters
from refactor.Pong.brain_shly import Actor, Critic
from refactor.Pong.hyper_parameters import Hyperparameters
from refactor.Pong.network import build_network

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main():
    # np.random.seed(2)
    # tf.set_random_seed(2)  # reproducible

    y_axis_ticks = [-25, -20, -15, -10, -5, 0]
    weights_path = './logs/weights/'
    data_path = './logs/data/'
    hp = Hyperparameters()

    gpu_options = tf.GPUOptions(allow_growth=True)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    env_name = 'Pong-v0'
    env = gym.make(env_name)
    env.seed(1)  # reproducible
    env = env.unwrapped

    net = build_network(n_features=hp.N_F, n_actions=hp.N_A, a_lr=hp.LR_A, c_lr=hp.LR_C, discount=hp.GAMMA)
    actor = Actor(sess, net[0])
    critic = Critic(sess, net[1])

    sess.run(tf.global_variables_initializer())

    if hp.OUTPUT_GRAPH:
        tf.summary.FileWriter("./logs/", sess.graph)

    episodes = []
    episode_rewards = []
    running_rewards = []
    total_steps = 0

    saver, load_episode = restore_parameters(sess, weights_path)
    # probs_path = data_path + 'probs_' + str(0) + '.txt'
    # td_exp_path = data_path + 'td_exp_' + str(0) + '.txt'
    # write_file(probs_path, 'probs\n', True)
    # write_file(td_exp_path, 'td_exp\n', True)

    for i_episode in range(hp.MAX_EPISODE):
        s = env.reset()
        s = preprocess_image(s, hp.N_F)
        # assert to check: whether there is nan in s.
        assert np.isnan(np.min(s.ravel())) == False

        episode_steps = 0
        track_r = []
        while True:
            # if hp.RENDER:
            #     env.render()

            # env.render()

            a, probs = actor.choose_action(s)
            probs = np.around(probs, decimals=4)
            # content = str([i_episode, total_steps]) + '  ' + str(probs.tolist()) + '\n'
            # write_file(probs_path, content, False)
            # print('------------------------------------', probs)

            s_, r, done, info = env.step(a)
            s_ = preprocess_image(s_, hp.N_F)
            # assert to check: whether there is nan in s_.
            assert np.isnan(np.min(s_.ravel())) == False

            if done:
                r = -2
            track_r.append(r)

            td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            exp_v = actor.learn(s, a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]

            s = s_
            episode_steps += 1
            total_steps += 1

            if done:
                # print(episode_steps)
                ep_rs_sum = sum(track_r)
                aa = track_r.count(1)

                if 'running_reward' not in globals() and 'running_reward' not in locals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05

                running_rewards.append(running_reward)
                # print(len(running_rewards))
                if len(running_rewards) % hp.SAVED_INTERVAL == 0:
                    # write_file(data_path + 'rewards_' + str(i_episode) + '.txt', running_rewards, True)
                    plot_rewards(running_rewards, y_axis_ticks, data_path)
                if i_episode % hp.SAVED_INTERVAL_NET == 0 and i_episode != 0:
                    save_parameters(sess, weights_path, saver,
                                    weights_path + '-' + str(load_episode + i_episode))
                print("episode: {0}, running reward: {1:.4f}, episode reward: {2}, td error: {3}, exp_v: {4}".
                      format(i_episode, running_reward, ep_rs_sum, td_error, exp_v))
                break


if __name__ == '__main__':
    main()
