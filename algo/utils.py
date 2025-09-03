import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Optional

from torch.nn.modules.dropout import Dropout
import torch
import numpy as np
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence

from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def convert_D4RL(self, dataset):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.reward = dataset['rewards'].reshape(-1,1)
        self.not_done = 1. - dataset['terminals'].reshape(-1,1)
        self.size = self.state.shape[0]


class MLP(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim,
        n_layers,
        activations: Callable = nn.ReLU,
        activate_final: int = False,
        dropout_rate: Optional[float] = None
    ) -> None:
        super().__init__()

        self.affines = []
        self.affines.append(nn.Linear(in_dim, hidden_dim))
        for i in range(n_layers-2):
            self.affines.append(nn.Linear(hidden_dim, hidden_dim))
        self.affines.append(nn.Linear(hidden_dim, out_dim))
        self.affines = nn.ModuleList(self.affines)

        self.activations = activations()
        self.activate_final = activate_final
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = Dropout(self.dropout_rate)
            self.norm_layer = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        for i in range(len(self.affines)):
            x = self.affines[i](x)
            if i != len(self.affines)-1 or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = self.dropout(x)
                    # x = self.norm_layer(x)
        return x

def identity(x):
    return x

def fanin_init(tensor, scale=1):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = scale / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

def orthogonal_init(tensor, gain=0.01):
    torch.nn.init.orthogonal_(tensor, gain=gain)

class ParallelizedLayerMLP(nn.Module):

    def __init__(
        self,
        ensemble_size,
        input_dim,
        output_dim,
        w_std_value=1.0,
        b_init_value=0.0
    ):
        super().__init__()

        # approximation to truncated normal of 2 stds
        w_init = torch.randn((ensemble_size, input_dim, output_dim))
        w_init = torch.fmod(w_init, 2) * w_std_value
        self.W = nn.Parameter(w_init, requires_grad=True)

        # constant initialization
        b_init = torch.zeros((ensemble_size, 1, output_dim)).float()
        b_init += b_init_value
        self.b = nn.Parameter(b_init, requires_grad=True)

    def forward(self, x):
        # assumes x is 3D: (ensemble_size, batch_size, dimension)
        return x @ self.W + self.b


class ParallelizedEnsembleFlattenMLP(nn.Module):

    def __init__(
            self,
            ensemble_size,
            hidden_sizes,
            input_size,
            output_size,
            init_w=3e-3,
            hidden_init=fanin_init,
            w_scale=1,
            b_init_value=0.1,
            layer_norm=None,
            final_init_scale=None,
            dropout_rate=None,
    ):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.input_size = input_size
        self.output_size = output_size
        self.elites = [i for i in range(self.ensemble_size)]

        self.sampler = np.random.default_rng()

        self.hidden_activation = F.relu
        self.output_activation = identity

        self.layer_norm = layer_norm

        self.fcs = []

        self.dropout_rate = dropout_rate
        if self.dropout_rate is not None:
            self.dropout = Dropout(self.dropout_rate)

        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = ParallelizedLayerMLP(
                ensemble_size=ensemble_size,
                input_dim=in_size,
                output_dim=next_size,
            )
            for j in self.elites:
                hidden_init(fc.W[j], w_scale)
                fc.b[j].data.fill_(b_init_value)
            self.__setattr__('fc%d'% i, fc)
            self.fcs.append(fc)
            in_size = next_size

        self.last_fc = ParallelizedLayerMLP(
            ensemble_size=ensemble_size,
            input_dim=in_size,
            output_dim=output_size,
        )
        if final_init_scale is None:
            self.last_fc.W.data.uniform_(-init_w, init_w)
            self.last_fc.b.data.uniform_(-init_w, init_w)
        else:
            for j in self.elites:
                orthogonal_init(self.last_fc.W[j], final_init_scale)
                self.last_fc.b[j].data.fill_(0)

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)

        state_dim = inputs[0].shape[-1]

        dim=len(flat_inputs.shape)
        # repeat h to make amenable to parallelization
        # if dim = 3, then we probably already did this somewhere else
        # (e.g. bootstrapping in training optimization)
        if dim < 3:
            flat_inputs = flat_inputs.unsqueeze(0)
            if dim == 1:
                flat_inputs = flat_inputs.unsqueeze(0)
            flat_inputs = flat_inputs.repeat(self.ensemble_size, 1, 1)

        # input normalization
        h = flat_inputs

        # standard feedforward network
        for _, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
            # add dropout
            if self.dropout_rate:
                h = self.dropout(h)
            if hasattr(self, 'layer_norm') and (self.layer_norm is not None):
                h = self.layer_norm(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)

        # if original dim was 1D, squeeze the extra created layer
        if dim == 1:
            output = output.squeeze(1)

        # output is (ensemble_size, batch_size, output_size)
        return output

    def sample(self, *inputs):
        preds = self.forward(*inputs)

        sample_idxs = np.random.choice(self.ensemble_size, 2, replace=False)
        preds_sample = preds[sample_idxs]

        return torch.min(preds_sample, dim=0)[0], sample_idxs

class KMeansStateNovelty:
    """
    An online novelty detector for continuous state spaces using Mini-Batch K-Means.

    A new state is considered novel if its distance to the nearest cluster centroid
    exceeds a dynamically calculated percentile-based threshold for that specific cluster.
    The model is updated incrementally using partial_fit.
    """

    def __init__(self, state_dim: int, n_clusters: int = 50, novelty_percentile: int = 90,
                 warmup_steps: int = 2000, update_threshold_freq: int = 1000):
        """
        Initializes the KMeansStateNovelty detector.

        Args:
            state_dim (int): The dimensionality of the state space.
            n_clusters (int): The number of clusters (k) to partition the state space into.
            novelty_percentile (int): The percentile of intra-cluster distances to use as the
                                      novelty threshold (e.g., 99 means a point is novel if it's
                                      further than 99% of existing points in its cluster).
            warmup_steps (int): The number of initial states to collect before the first
                                model fit. All states are considered novel during this phase.
            update_threshold_freq (int): How often (in steps) to recalculate the novelty
                                         thresholds for each cluster.
        """
        if not (0 < novelty_percentile < 100):
            raise ValueError("novelty_percentile must be between 0 and 100.")

        self.state_dim = state_dim
        self.n_clusters = n_clusters
        self.novelty_percentile = novelty_percentile
        self.warmup_steps = warmup_steps
        self.update_threshold_freq = update_threshold_freq

        self.model = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            n_init=10,  # Use multiple initializations for the first fit
            batch_size=256,
            max_iter=100
        )

        self.warmup_buffer = []
        self.is_warmed_up = False
        self.steps_since_threshold_update = 0

        # Data structures for online threshold calculation
        self.cluster_thresholds = np.zeros(self.n_clusters)
        self.cluster_distances = defaultdict(list)

    def _calculate_thresholds(self):
        """Calculates the novelty threshold for each cluster based on stored distances."""
        for i in range(self.n_clusters):
            if self.cluster_distances[i]:
                self.cluster_thresholds[i] = np.percentile(
                    self.cluster_distances[i], self.novelty_percentile
                )
            else:
                # If a cluster has no points, set a very high threshold
                self.cluster_thresholds[i] = np.inf

    def check_and_update(self, state: np.ndarray) -> bool:
        """
        Checks if a new state is novel and updates the internal model.

        Args:
            state (np.ndarray): The new state vector from the environment.

        Returns:
            bool: True if the state is considered novel, False otherwise.
        """
        if state.shape[0] != self.state_dim:
            raise ValueError(f"Input state dimension {state.shape[0]} does not match initialized dimension {self.state_dim}")

        state_reshaped = state.reshape(1, -1)

        # --- Warm-up Phase ---
        if not self.is_warmed_up:
            self.warmup_buffer.append(state)
            if len(self.warmup_buffer) >= self.warmup_steps:
                # First fit on the collected data
                warmup_data = np.array(self.warmup_buffer)
                self.model.fit(warmup_data)

                # Initial threshold calculation
                labels = self.model.predict(warmup_data)
                distances = np.linalg.norm(warmup_data - self.model.cluster_centers_[labels], axis=1)
                for i, dist in enumerate(distances):
                    self.cluster_distances[labels[i]].append(dist)
                self._calculate_thresholds()

                self.is_warmed_up = True
                self.warmup_buffer = [] # Clear buffer to save memory
            return True

        # --- Online Phase ---
        # Predict the cluster for the new state
        cluster_idx = self.model.predict(state_reshaped)[0]

        # Calculate distance to the assigned centroid
        centroid = self.model.cluster_centers_[cluster_idx]
        distance = np.linalg.norm(state - centroid)

        # Check for novelty
        is_novel = distance > self.cluster_thresholds[cluster_idx]

        # Update the model and statistics
        self.model.partial_fit(state_reshaped)
        self.cluster_distances[cluster_idx].append(distance)

        self.steps_since_threshold_update += 1
        if self.steps_since_threshold_update >= self.update_threshold_freq:
            self._calculate_thresholds()
            self.steps_since_threshold_update = 0

        return is_novel