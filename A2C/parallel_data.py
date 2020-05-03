import multiprocessing as mp
from collections import namedtuple
import gym
gym.logger.set_level(40)

class parallelenv():

    def __init__(self, _env, _number_of_workers):
        self.env = _env
        self.number_of_workers = _number_of_workers

    def worker(self, child_pipe):
        
