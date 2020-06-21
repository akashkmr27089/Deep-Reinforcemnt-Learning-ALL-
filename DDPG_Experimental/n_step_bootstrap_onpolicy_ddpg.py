import numpy as np
import matplotlib.pyplot as plt
from model import Actor, Critic, hidden_init
import gym
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
env = gym.make("MountainCarContinuous-v0")
action_size = env.action_space.shape[0]
state_size = env.observation_space.shape[0]

LR_CRITIC = 0.001
LR_ACTOR = 0.001
TAU = 1e-3
GAMMA = 0.99

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class Agent():
    def __init__(self, state_size, action_size, seed):
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.qvalue_local = Critic(state_size, action_size, seed).to(device)
        self.qvalue_target = Critic(state_size, action_size, seed).to(device)
        self.seed = seed


        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.qvalue_target.load_state_dict(self.qvalue_local.state_dict())
        self.noise = OUNoise(action_size, self.seed)

        self.qvalue_optimizer = optim.Adam(self.qvalue_local.parameters(), lr=LR_CRITIC)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

    def reset(self):
        self.actor_local.reset_parameters()
        self.qvalue_local.reset_parameters()
        self.actor_target.reset_parameters()
        self.qvalue_target.reset_parameters()
        self.noise.reset()

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def into_tensor(self, state):
        return torch.tensor(state).float().to(device)

    def play(self, max_len = 200):
        state1 = env.reset()
        score = 0
        for _ in range(max_len):
            state1 = self.into_tensor(state1)
            action1 = self.act(state)
            action1_t = self.into_tensor(action1)
            value1_pred = self.qvalue_local(state1, action1_t)

            state2, reward, done, _ = env.step(action1)

            state2 = self.into_tensor(state2)
            action2 = self.actor_target(state2)
            action2_t = self.into_tensor(action2)
            value2 = self.qvalue_target(state2, action2_t)

            if not done:
                expected_value = self.into_tensor(reward) + GAMMA * value2
            else:
                expected_value = self.into_tensor(reward)
            score += reward
            ## Critic Update
            qvalue_loss = F.mse_loss(value1_pred, expected_value)
            self.qvalue_optimizer.zero_grad()
            qvalue_loss.backward()
            self.qvalue_optimizer.step()

            ## Actor Update
            actor_loss = -self.qvalue_target(state1, action1_t)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor_local, self.actor_target, TAU)
            self.soft_update(self.qvalue_local, self.qvalue_target, TAU) 
            if done: break
            # print(actor_loss, qvalue_loss)
        return score

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def run(self, iteration, max_len = 200):
        self.score_deque = deque(maxlen=100)
        for episode in range(iteration):
            score = self.play(max_len)
            self.score_deque.append(score)
            print("\rIteration {} with Current Score {}".format(episode, score), end=" ")
            if(episode%50 == 0):
                print("\rIteration {} with Average Score {} ".format(episode, sum(self.score_deque)/len(self.score_deque)))