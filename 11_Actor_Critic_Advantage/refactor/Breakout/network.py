import numpy as np
import tensorflow as tf


def build_actor_network(n_features, n_actions, lr):
    s = tf.placeholder(tf.float32, [1, n_features, n_features], "state")
    a = tf.placeholder(tf.int32, None, "act")
    td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

    with tf.variable_scope('Actor'):
        input_crop = s / 255
        input = tf.transpose(input_crop, [1, 2, 0])
        conv1 = tf.contrib.layers.conv2d(inputs=input[np.newaxis, :], num_outputs=32, kernel_size=8, stride=4)
        conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=32, kernel_size=4, stride=2)
        conv3 = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=32, kernel_size=3, stride=1)

        flat = tf.contrib.layers.flatten(conv3)
        f = tf.contrib.layers.fully_connected(flat, 512)
        acts_prob = tf.contrib.layers.fully_connected(f, n_actions, activation_fn=tf.nn.softmax)

    with tf.variable_scope('exp_v'):
        log_prob = tf.log(acts_prob[0, a])
        exp_v = tf.reduce_mean(log_prob * td_error)  # advantage (TD_error) guided loss

    with tf.variable_scope('train'):
        train_op = tf.train.AdamOptimizer(lr).minimize(-exp_v)  # minimize(-exp_v) = maximize(exp_v)

    return [[s, a, td_error], [acts_prob, exp_v, train_op]]


def build_critic_network(n_features, lr, discount):
    s = tf.placeholder(tf.float32, [1, n_features, n_features], "state")
    v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
    r = tf.placeholder(tf.float32, None, 'r')

    with tf.variable_scope('Critic'):
        input_crop = s / 255
        input = tf.transpose(input_crop, [1, 2, 0])
        conv1 = tf.contrib.layers.conv2d(inputs=input[np.newaxis, :], num_outputs=32, kernel_size=8, stride=4)
        conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=32, kernel_size=4, stride=2)
        conv3 = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=32, kernel_size=3, stride=1)

        flat = tf.contrib.layers.flatten(conv3)
        f = tf.contrib.layers.fully_connected(flat, 512)
        v = tf.contrib.layers.fully_connected(f, 1, activation_fn=None)

    with tf.variable_scope('squared_TD_error'):
        td_error = r + discount * v_ - v
        loss = tf.square(td_error)  # TD_error = (r+gamma*V_next) - V_eval
    with tf.variable_scope('train'):
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    return [[s, v_, r], [v, td_error, loss, train_op]]


def build_actor_ram_network(n_features, n_actions, lr):
    s = tf.placeholder(tf.float32, [1, n_features], "state")
    a = tf.placeholder(tf.int32, None, "act")
    td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

    with tf.variable_scope('Actor'):
        input = s / 255
        l1 = tf.contrib.layers.fully_connected(input, 20, activation_fn=tf.nn.relu)
        acts_prob = tf.contrib.layers.fully_connected(l1, n_actions, activation_fn=tf.nn.softmax)

    with tf.variable_scope('exp_v'):
        log_prob = tf.log(tf.clip_by_value(acts_prob, 1e-2, 1)[0, a])
        exp_v = tf.reduce_mean(log_prob * td_error)  # advantage (TD_error) guided loss

    with tf.variable_scope('train'):
        train_op = tf.train.AdamOptimizer(lr).minimize(-exp_v)  # minimize(-exp_v) = maximize(exp_v)

    return [[s, a, td_error], [acts_prob, exp_v, train_op]]


def build_critic_ram_network(n_features, lr, discount):
    s = tf.placeholder(tf.float32, [1, n_features], "state")
    v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
    r = tf.placeholder(tf.float32, None, 'r')

    with tf.variable_scope('Critic'):
        input = s / 255
        l1 = tf.contrib.layers.fully_connected(input, 20, activation_fn=tf.nn.relu)
        v = tf.contrib.layers.fully_connected(l1, 1, activation_fn=None)

    with tf.variable_scope('squared_TD_error'):
        td_error = r + discount * v_ - v
        loss = tf.square(td_error)  # TD_error = (r+gamma*V_next) - V_eval
    with tf.variable_scope('train'):
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    return [[s, v_, r], [v, td_error, loss, train_op]]
