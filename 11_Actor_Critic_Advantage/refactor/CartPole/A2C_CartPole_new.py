"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.
The cart pole example. Policy is oscillated.
"""

import gym
import numpy as np
import tensorflow as tf

from utils import write_file, plot_rewards
from refactor.CartPole.brain import Actor, Critic
from refactor.CartPole.hyper_parameters import Hyperparameters


def main():
    np.random.seed(2)
    tf.set_random_seed(2)  # reproducible

    sess = tf.Session()
    hp = Hyperparameters()

    env = gym.make('CartPole-v0')
    env.seed(1)  # reproducible
    # env = env.unwrapped

    actor = Actor(sess, n_features=hp.N_F, n_actions=hp.N_A, lr=hp.LR_A)
    # we need a good teacher, so the teacher should learn faster than the actor
    critic = Critic(sess, n_features=hp.N_F, lr=hp.LR_C, discount=hp.GAMMA)

    sess.run(tf.global_variables_initializer())

    if hp.OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/", sess.graph)

    running_rewards = []

    for i_episode in range(hp.MAX_EPISODE):
        s = env.reset()
        t = 0
        track_r = []
        while True:
            if hp.RENDER:
                env.render()

            a, probs = actor.choose_action(s)
            if i_episode == 0:
                write_file('./logs/probs.txt', probs, True)
            else:
                write_file('./logs/probs.txt', probs, False)
            # print('------------------------------------', probs)

            s_, r, done, info = env.step(a)

            if done:
                r = -20

            track_r.append(r)

            td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            exp_v = actor.learn(s, a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]

            s = s_
            t += 1

            if done or t >= hp.MAX_EP_STEPS:
                ep_rs_sum = sum(track_r)

                if 'running_reward' not in globals() and 'running_reward' not in locals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05

                running_rewards.append(running_reward)
                # print(len(running_rewards))
                if len(running_rewards) % 1000 == 0:
                    write_file('./logs/Test/rewards_' + str(i_episode) + '.txt', running_rewards, True)
                    y_axis_ticks = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
                    plot_rewards(running_rewards, y_axis_ticks, './logs/Test/' + str(i_episode) + '/')
                if running_reward > hp.DISPLAY_REWARD_THRESHOLD:
                    hp.RENDER = True  # rendering
                # print("episode:", i_episode, "  reward:", int(running_reward))
                print('episode:', i_episode, ' running reward:', int(running_reward),
                      ' episode reward:', ep_rs_sum, ' tf_error:', td_error, ' exp_v:', exp_v)
                break


if __name__ == '__main__':
    main()
