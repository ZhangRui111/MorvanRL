"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.
The cart pole example. Policy is oscillated.
"""

import gym
import numpy as np
import tensorflow as tf

from utils import write_file, plot_rewards
from actor_critic import Actor, Critic


def main():
    np.random.seed(5)
    tf.set_random_seed(2)  # reproducible

    OUTPUT_GRAPH = False
    MAX_EPISODE = 2001
    DISPLAY_REWARD_THRESHOLD = 10000  # renders environment if total episode reward is greater then this threshold
    MAX_EP_STEPS = 1000  # maximum time step in one episode
    RENDER = True  # rendering wastes time
    GAMMA = 0.9  # reward discount in TD error
    LR_A = 0.01  # 0.01: learning rate for actor
    LR_C = 0.1  # 0.1: learning rate for critic

    sess = tf.Session()

    env = gym.make('CartPole-v0')
    env.seed(1)  # reproducible
    env = env.unwrapped

    N_F = env.observation_space.shape[0]
    N_A = env.action_space.n

    actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
    # we need a good teacher, so the teacher should learn faster than the actor
    critic = Critic(sess, n_features=N_F, lr=LR_C)

    sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/Test/", sess.graph)

    running_rewards = []

    for i_episode in range(MAX_EPISODE):
        s = env.reset()
        t = 0
        track_r = []
        while True:
            if RENDER:
                env.render()

            a, probs = actor.choose_action(s)
            if i_episode == 0:
                write_file('./logs/Test/probs.txt', probs, True)
            else:
                write_file('./logs/Test/probs.txt', probs, False)
            # print('------------------------------------', probs)

            s_, r, done, info = env.step(a)

            if done:
                r = -20

            track_r.append(r)

            td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(s, a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]

            s = s_
            t += 1

            if done or t >= MAX_EP_STEPS:
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
                if running_reward > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True  # rendering
                # print("episode:", i_episode, "  reward:", int(running_reward))
                print("ep: %d, running_reward: %f, ep_rs_sum: %f" % (i_episode, running_reward, ep_rs_sum))
                break


if __name__ == '__main__':
    main()
