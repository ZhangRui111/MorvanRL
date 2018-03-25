"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list ['u', 'd', 'l', 'r']
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have same value.
            # # DataFrame.idxmax(axis=0, skipna=True)
            # Return index of first occurrence of maximum over requested axis. So if there are some
            # actions with same value, we need to shuffle all actions before we choose the idxmax action.
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            # # numpy.random.permutation(x)
            # # DataFrame.reindex(......)
            # Conform DataFrame to new index with optional filling logic, placing NA/NaN in locations having
            # no value in the previous index. A new object is produced unless the new index is equivalent to
            # the current one and copy=False
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

# # numpy.random.permutation(x)
# Randomly permute a sequence, or return a permuted range.
# If x is a multi-dimensional array, it is only shuffled along its first index.
#
# Parameters:
# x : int or array_like
# If x is an integer, randomly permute np.arange(x). If x is an array, make a copy and shuffle
# the elements randomly.

# >>> np.random.permutation(10)
# array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6])
# >>> np.random.permutation([1, 4, 9, 12, 15])
# array([15,  1,  9,  4, 12])
# >>> arr = np.arange(9).reshape((3, 3))
# >>> np.random.permutation(arr)
# array([[6, 7, 8],
#        [0, 1, 2],
#        [3, 4, 5]])
