import numpy as np


class Actor(object):
    def __init__(self, sess, net):
        self.sess = sess
        self.s = net[0][0]
        self.a = net[0][1]
        self.td_error = net[0][2]
        self.acts_prob = net[1][0]
        self.exp_v = net[1][1]
        self.train_op = net[1][2]

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})  # get probabilities for all actions
        assert np.isnan(np.min(probs.ravel())) == False
        action = np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())
        return action, probs.ravel()


class Critic(object):
    def __init__(self, sess, net):
        self.sess = sess
        self.s = net[0][0]
        self.v_ = net[0][1]
        self.r = net[0][2]
        self.v = net[1][0]
        self.td_error = net[1][1]
        self.loss = net[1][2]
        self.train_op = net[1][3]

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    feed_dict={self.s: s, self.v_: v_, self.r: r})
        return td_error
