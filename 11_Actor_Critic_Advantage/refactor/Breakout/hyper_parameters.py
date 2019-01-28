class Hyperparameters(object):
    def __init__(self):
        self.model = 'A2C_Q'

        self.OUTPUT_GRAPH = False
        self.MAX_EPISODE = 500001
        self.N_F = 80
        self.N_A = 4
        self.DISPLAY_REWARD_THRESHOLD = 0  # renders = True, if total episode reward is greater then this threshold.
        self.MAX_EP_STEPS = 600
        self.SAVED_INTERVAL = 500
        self.SAVED_INTERVAL_NET = 5000
        self.RENDER = False  # rendering wastes time
        self.GAMMA = 0.9  # reward discount in TD error
        self.LR_A = 0.00001  # 0.01: learning rate for actor
        self.LR_C = 0.0001  # 0.1: learning rate for critic
