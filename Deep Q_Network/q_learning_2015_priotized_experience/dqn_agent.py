import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e3)  # replay buffer size
BATCH_SIZE = int(1e2)       # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4      # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def single_loss(self, state, action, reward, next_state, done):
            max_val = self.qnetwork_target(torch.from_numpy(next_state).to(device))
            max_val = max(max_val)
            target = reward + GAMMA*max_val*(1-done)
            current = self.qnetwork_local(torch.from_numpy(state).to(device))[action]
            loss = float(F.mse_loss(target, current))
            return loss
        
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        loss = self.single_loss(state, action, reward, next_state, done)
        self.memory.add(state, action, reward, next_state, done, loss)
        
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) == BUFFER_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=1.0):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()) #expliding
        else:
            return random.choice(np.arange(self.action_size))  #exploration

    def learn(self, experiences, gamma):

        states, actions, rewards, next_states, dones, _ = experiences
        ## TODO: compute and minimize the loss
        max_val = self.qnetwork_target(next_states)
        best_val = max_val.argmax(dim= -1)
        max_val = max_val.gather(-1, best_val.reshape(max_val.size()[0],1))
        target = rewards + gamma*max_val*(1-dones)
        current = self.qnetwork_local(states).gather(-1, actions.reshape(actions.size()[0],1))
        # adding line to add losses
        for i in range(len(states)):
            loss2 = float(F.mse_loss(current[i], target[i]))
            # if float(dones[i].cpu()) == 1.0:
            #     print(loss2, (1-float(dones[i].cpu())))
            self.memory.add(states[i].cpu(), actions[i].cpu(), rewards[i].cpu(), next_states[i].cpu(), dones[i].cpu(), loss2*(1-float(dones[i].cpu())))
        #edit over
        self.loss = F.mse_loss(current, target)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):

        self.action_size = action_size
        self.memory = []  
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "losses"])
        self.seed = random.seed(seed)
        self.temp = 0
    
    def add(self, state, action, reward, next_state, done, loss):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, loss)
        self.memory.append(e)
        if len(self.memory) > self.buffer_size:
            del self.memory[0]


    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        #experiences = random.sample(self.memory, k=self.batch_size)
        _, _, _,_, _,losses = zip(*self.memory)
        temp = sum(losses)
        prob_list = [x/temp for x in losses]
        selected_list = np.random.choice(np.arange(len(prob_list)), self.batch_size, p=prob_list, replace = False)
        selected_list = list(selected_list)
        selected_list.sort(reverse=True)
        experiences = [self.memory.pop(x) for x in selected_list]
        #experiences = [self.memory.pop(random.randrange(len(self.memory))) for _ in range(self.batch_size)]
        # adding lines to delete experience
        #end of edit

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        losses = torch.from_numpy(np.vstack([e.losses for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        self.temp = losses
        return (states, actions, rewards, next_states, dones, losses)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
