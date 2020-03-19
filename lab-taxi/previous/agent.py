import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, epsilon = 1.0, eps_start=0.01, eps_decay=0.9999):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.epsilon = epsilon

    def q_pros(self, Q_state):
        policy = np.ones(self.nA)*(self.epsilon/self.nA)
        best_arg = np.argmax(Q_state)
        policy[best_arg] = (1 - self.epsilon) + (self.epsilon/self.nA)
        return policy

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.epsilon = max(self.epsilon*self.eps_decay, self.eps_start)
        # print(self.epsilon)
        action = np.random.choice(np.arange(self.nA), p=self.q_pros(state)) if state in self.Q else np.random.randint(self.nA)
        return action
        """
        return np.argmax(state)
        """

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        """
        self.Q[state][action] = self.Q[state][action] + alpha*(reward + gamma*self.Q[next_state][np.argmax(self.Q[next_state])] \
            - self.Q[state][action])

        """
        #self.Q[state][action] = self.Q[state][action] + alpha*(reward + gamma*self.Q[next_state][np.random.choice(np.arange(self.nA), p=self.q_pros(state))] - self.Q[state][action])
        self.Q[state][action] = self.Q[state][action] + alpha*(reward + gamma*self.Q[next_state][np.argmax(Q[next_state])] - self.Q[state][action])