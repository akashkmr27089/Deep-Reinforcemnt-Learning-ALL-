import numpy as np
from collections import defaultdict
import random

class Agent:

    def __init__(self, epsilon, alpha, gamma, eps_start):
        self.nA = 6
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps_start = eps_start
        self.eps_decay = 0.9999
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def q_pros(self, Q_state):
        policy = np.ones(self.nA)*(self.epsilon/self.nA)
        best_arg = np.argmax(Q_state)
        policy[best_arg] = (1 - self.epsilon) + (self.epsilon/self.nA)
        return policy

    def select_action(self, state):
        #self.epsilon = max(self.epsilon*self.eps_decay, self.eps_start)
        # print(self.epsilon)
        #action = np.random.choice(np.arange(self.nA), p=self.q_pros(state)) if state in self.Q else np.random.randint(self.nA)
        #return action
        if random.uniform(0,1) < self.epsilon:
            return random.randint(0,5)
        else:
            return np.argmax(self.Q[state])

    def step(self, state, action, reward, next_state, done, alpha = 0.1, gamma=0.5):
        #self.Q[state][action] = self.Q[state][action] + self.alpha*(reward + self.gamma*self.Q[next_state][np.random.choice(np.arange(self.nA), p=self.q_pros(state))] \
        #    - self.Q[state][action])
        if not done:
            new_value = self.Q[next_state][np.random.choice(np.arange(self.nA), p=self.q_pros(self.Q[next_state]))]
            self.Q[state][action] = self.Q[state][action] + self.alpha*(reward + self.gamma*new_value -self.Q[state][action])
        else:
            self.epsilon = max(self.epsilon*self.eps_decay, self.eps_start)
            self.Q[state][action] = self.Q[state][action] + self.alpha*(reward - self.Q[state][action])