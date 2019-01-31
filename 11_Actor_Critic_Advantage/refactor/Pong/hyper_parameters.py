class Hyperparameters(object):
    def __init__(self):
        self.model = 'A2C_Q'

        self.OUTPUT_GRAPH = False
        self.MAX_EPISODE = 300001
        self.N_F = 80
        self.N_A = 6
        self.DISPLAY_REWARD_THRESHOLD = -25  # renders = True, if total episode reward is greater then this threshold.
        self.MAX_EP_STEPS = 600
        self.SAVED_INTERVAL = 1000
        self.SAVED_INTERVAL_NET = 3000
        self.RENDER = False  # rendering wastes time
        self.GAMMA = 0.99  # reward discount in TD error
        self.LR_A = 0.00001  # learning rate for actor
        self.LR_C = 0.0001  # learning rate for critic
