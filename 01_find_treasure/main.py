"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.

View more on Morvan's tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
import time

np.random.seed(2)  # reproducible


N_STATES = 6   # the length of the 1 dimensional world
ACTIONS = ['left', 'right']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 15   # maximum episodes
FRESH_TIME = 0.2    # fresh time for one move


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    # print(table)    # show table
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    # # DataFrame.iloc
    # act non-greedy or state-action have no value
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):
        # # a = (state_actions == 0)
        # a:
        # left      True
        # right    False
        # Name: 0, dtype: bool
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.idxmax()
        # replace argmax to idxmax as argmax means a different function in newer version of pandas.
        # # DataFrame.idxmax(axis=0, skipna=True)

    return action_name


def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'right':    # move right
        if S == N_STATES - 2:   # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:   # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:

            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)  # take action & get next state and reward
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
            else:
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update

            S = S_  # move to next state
            update_env(S, episode, step_counter+1)
            step_counter += 1

    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)


# # DataFrame.iloc
# Purely integer-location based indexing for selection by position.
# .iloc[] is primarily integer position based (from 0 to length-1 of the axis), but may also be used
# with a boolean array.
#
# Allowed inputs are:
#     An integer, e.g. 5.
#     A list or array of integers, e.g. [4, 3, 0].
#     A slice object with ints, e.g. 1:7.
#     A boolean array.
#
# In [58]: df1
# Out[58]:
#            0         2         4         6
# 0   0.149748 -0.732339  0.687738  0.176444
# 2   0.403310 -0.154951  0.301624 -2.179861
# 4  -1.369849 -0.954208  1.462696 -1.743161
# 6  -0.826591 -0.345352  1.314232  0.690579
# 8   0.995761  2.396780  0.014871  3.357427
# 10 -0.317441 -1.236269  0.896171 -0.487602
# In [59]: df1.iloc[:3]
# Out[59]:
#           0         2         4         6
# 0  0.149748 -0.732339  0.687738  0.176444
# 2  0.403310 -0.154951  0.301624 -2.179861
# 4 -1.369849 -0.954208  1.462696 -1.743161
#
# In [60]: df1.iloc[1:5,2:4]
# Out[60]:
#           4         6
# 2  0.301624 -2.179861
# 4  1.462696 -1.743161
# 6  1.314232  0.690579
# 8  0.014871  3.357427

# # DataFrame.idxmax(axis=0, skipna=True)
# Return index of first occurrence of maximum over requested axis. NA/null values are excluded.
# About idxmax&argmax refer to:https://www.jianshu.com/p/f21f01a92521
