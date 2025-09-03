import torch
import torch.nn as nn

class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x):
        return self.network(x)

class MetricModel(nn.Module):
    """
    Approximates the inverse dynamic bisimulation metric d(s_i, s_j).
    This model takes two states and outputs a scalar distance.
    This is used to compute the potential Phi(s) = d(s, s_0).
    """
    def __init__(self, state_dim, hidden_size=256):
        super(MetricModel, self).__init__()
        # Input will be two concatenated states
        self.network = MLPNetwork(state_dim * 2, 1, hidden_size)

    def forward(self, state1, state2):
        x = torch.cat([state1, state2], dim=-1)
        return self.network(x)

class InverseDynamicsModel(nn.Module):
    """
    Predicts the action that caused the transition from state s_t to s_{t+1}.
    """
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(InverseDynamicsModel, self).__init__()
        # Input is two concatenated states
        self.network = MLPNetwork(state_dim * 2, action_dim, hidden_size)

    def forward(self, state, next_state):
        x = torch.cat([state, next_state], dim=-1)
        return self.network(x)

class ForwardDynamicsModel(nn.Module):
    """
    Predicts the next state given the current state and action.
    Outputs parameters for a Gaussian distribution (mu, log_std).
    """
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(ForwardDynamicsModel, self).__init__()
        # Input is state and action concatenated
        # Output is mu and log_std for the next state distribution
        self.network = MLPNetwork(state_dim + action_dim, state_dim * 2, hidden_size)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        mu, log_std = self.network(x).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -5, 2) # Clamp for stability
        return mu, log_std.exp()
