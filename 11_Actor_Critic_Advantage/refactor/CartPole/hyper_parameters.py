

class Hyperparameters(object):
    def __init__(self):
        self.model = 'A2C_Q'

        self.OUTPUT_GRAPH = False
        self.MAX_EPISODE = 2001
        self.N_F = 4
        self.N_A = 2
        # renders environment if total episode reward is greater then this threshold
        self.DISPLAY_REWARD_THRESHOLD = 10000
        self.MAX_EP_STEPS = 1000  # maximum time step in one episode
        self.RENDER = False  # rendering wastes time
        self.GAMMA = 0.9  # reward discount in TD error
        self.LR_A = 0.01  # 0.01: learning rate for actor
        self.LR_C = 0.1  # 0.1: learning rate for critic
