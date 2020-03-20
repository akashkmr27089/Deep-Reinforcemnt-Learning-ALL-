from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v3')
agent = Agent(alpha = 0.4762 , eps_start = 0.01268, epsilon =  0.04973  , gamma = 0.9662)
avg_rewards, best_avg_reward = interact(env, agent)