import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):  
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_dim = state_size
        self.output_dim = action_size
        "*** YOUR CODE HERE ***"

        self.feature_layer = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.ReLU(),
	    nn.Linear(32, 32),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32,1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32, self.output_dim)
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())
        return qvals
