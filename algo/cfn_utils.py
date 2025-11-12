import numpy as np
import random
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

class CoinFlipNetwork(nn.Module):
    def __init__(self, state_dim, d=20):
        """
        state_dim : dimensionality of state input
        d         : number of coin flips (output dim)
        """
        super().__init__()
        self.d = d

        # Trainable component
        self.f_hat = MLPNetwork(state_dim, d)

        # Frozen random prior component
        self.f_prior = MLPNetwork(state_dim, d)
        for p in self.f_prior.parameters():
            p.requires_grad = False
        self.f_prior.eval()  # freeze dropout/bn if present

        # Running mean and variance of prior output
        self.register_buffer("running_mean", torch.zeros(d))
        self.register_buffer("running_var", torch.ones(d))
        self.momentum = 0.99  # update rate for running stats

    @torch.no_grad()
    def update_prior_stats(self, prior_output):
        """Update running mean and variance for the frozen prior."""
        batch_mean = prior_output.mean(0)
        batch_var = prior_output.var(0, unbiased=False)

        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

    def normalize_prior(self, prior_output):
        """Normalize prior output using running stats."""
        return (prior_output - self.running_mean) / torch.sqrt(self.running_var + 1e-8)

    def forward(self, states):
        # Forward both networks
        with torch.no_grad():
            prior_raw = self.f_prior(states)
            self.update_prior_stats(prior_raw)  # keep stats up to date

        prior_normed = self.normalize_prior(prior_raw)
        trainable_out = self.f_hat(states)

        # Total CFN output = trainable + normalized frozen prior
        total = trainable_out + prior_normed
        return total

    def intrinsic_bonus(self, states):
        """Compute intrinsic reward (B(s)) = sqrt(1/d * ||f(s)||^2)."""
        f_out = self.forward(states).detach()
        bonus = torch.sqrt((f_out.pow(2).sum(1) / self.d) + 1e-8)
        return bonus

class SumTree:
    """
    A SumTree data structure used for prioritized experience replay.

    This implementation is based on the one found in OpenAI's baselines:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

    The tree is stored in a simple 1D numpy array.

    Tree structure:
    [ ... internal nodes ... ] [ ... leaf nodes (priorities) ... ]
    0 ... (capacity - 2)     (capacity - 1) ... (2 * capacity - 2)

    The data (transitions) is stored in a separate array.
    """

    def __init__(self, capacity):
        """
        Initialize the SumTree.

        Args:
            capacity (int): The maximum number of items to store in the buffer.
                            Must be a power of 2.
        """
        if not ((capacity & (capacity - 1) == 0) and capacity > 0):
            # Find the next power of 2
            import math
            next_pow_2 = 2**math.ceil(math.log2(capacity))
            print(f"Warning: SumTree capacity must be a power of 2. "
                  f"Rounding up {capacity} to {next_pow_2}.")
            capacity = next_pow_2

        self.capacity = capacity
        # Total nodes = 2 * capacity - 1
        # We use a 1D array to store the tree
        # Internal nodes are at indices 0 to (capacity - 2)
        # Leaf nodes (priorities) are at indices (capacity - 1) to (2 * capacity - 2)
        self.tree = np.zeros(2 * capacity - 1)

        # The actual data (e.g., transitions)
        # Each item will be: ((state, ...), n_samples)
        self.data = np.zeros(capacity, dtype=object)

        # Current number of entries in the buffer
        self.n_entries = 0

        # Pointer to the next data write location
        self.write_idx = 0

    def _propagate(self, tree_idx, change):
        """Propagate a change in priority up the tree."""
        parent_idx = (tree_idx - 1) // 2
        self.tree[parent_idx] += change

        # Recursively propagate until we reach the root
        if parent_idx != 0:
            self._propagate(parent_idx, change)

    def update(self, tree_idx, priority):
        """
        Update the priority of a leaf node and propagate the change.

        Args:
            tree_idx (int): The index in the tree array (2*cap - 1) of the leaf.
            priority (float): The new priority value.
        """
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def add(self, priority, data):
        """
        Add a new experience to the buffer.

        Args:
            priority (float): The priority of the new experience.
            data (object): The experience data (e.g., a transition tuple).
        """
        # Get the tree index for the leaf node
        tree_idx = self.write_idx + self.capacity - 1

        # Store the data
        self.data[self.write_idx] = data

        # Update the tree with the new priority
        self.update(tree_idx, priority)

        # Move the write pointer
        self.write_idx = (self.write_idx + 1) % self.capacity

        # Update the total number of entries
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def total_priority(self):
        """Return the total priority (value of the root node)."""
        return self.tree[0]

    def get(self, s):
        """
        Retrieve a leaf node (priority and data) based on a sample value 's'.

        This performs a binary search-like operation down the tree.

        Args:
            s (float): A random sample value between 0 and total_priority.

        Returns:
            tuple: (tree_idx, priority, data)
        """
        tree_idx = 0  # Start at the root

        while True:
            left_child_idx = 2 * tree_idx + 1
            right_child_idx = left_child_idx + 1

            # If we are at a leaf node
            if left_child_idx >= len(self.tree):
                break

            # Follow the path down the tree
            if s <= self.tree[left_child_idx]:
                tree_idx = left_child_idx
            else:
                s -= self.tree[left_child_idx]
                tree_idx = right_child_idx

        # The data index is offset from the tree index
        data_idx = tree_idx - self.capacity + 1

        return (tree_idx, self.tree[tree_idx], self.data[data_idx])

    def get_data_idx(self, tree_idx):
        """Converts a tree leaf index to a data index."""
        return tree_idx - self.capacity + 1

    def update_data(self, data_idx, new_data):
        """Updates the data at a specific data index."""
        if data_idx < 0 or data_idx >= self.capacity:
            raise IndexError("Data index out of range")
        self.data[data_idx] = new_data

    def __len__(self):
        return self.n_entries

class CFNReplayBuffer:

    def __init__(self, capacity, device):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.device = device

    def add(self, priority, state, coin_flips):
        # Ensure data is stored as numpy arrays for consistency
        experience = (np.array(state), coin_flips.cpu().numpy())
        data = (experience, 0)  # n_samples initialized to 0

        self.tree.add(priority, data)

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            tuple:
                - batch (tuple): (states, actions, rewards, next_states, dones)
                - priorities (np.array): The priorities of the sampled items.
                - n_samples (np.array): The number of times each item has been sampled.
                - tree_indices (np.array): Tree indices for updating priorities later.
        """
        batch_data = []
        tree_indices = np.empty(batch_size, dtype=int)
        batch_n_samples = np.empty(batch_size, dtype=int)

        # Calculate priority segment size
        total_p = self.tree.total_priority()

        if self.tree.n_entries < batch_size:
            raise ValueError(f"Not enough samples in buffer. "
                             f"Requested {batch_size}, but only {self.tree.n_entries} available.")

        if total_p == 0:
            raise RuntimeError("Cannot sample from buffer with total priority 0.")

        segment = total_p / batch_size

        for i in range(batch_size):
            # Sample a value from each segment
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            # Retrieve the corresponding experience
            (tree_idx, _, data) = self.tree.get(s)

            # Unpack the stored data
            (experience, n_samples) = data

            # Increment n_samples count
            n_samples += 1

            # Update the data in the SumTree (to save the new n_samples)
            data_idx = self.tree.get_data_idx(tree_idx)
            self.tree.update_data(data_idx, (experience, n_samples))

            # Store data for the batch
            batch_data.append(experience)
            tree_indices[i] = tree_idx
            batch_n_samples[i] = n_samples


        # Unzip the batch data
        states, coin_flips = zip(*batch_data)

        batch = (
            torch.FloatTensor(np.array(states)).to(self.device),
            torch.FloatTensor(np.array(coin_flips)).to(self.device),
        )

        return batch, batch_n_samples, tree_indices

    def update_priorities(self, tree_indices, priorities):
        """
        Update the priorities of sampled experiences.

        Args:
            tree_indices (np.array): Array of tree indices from the sample() call.
            priorities (np.array): Array of new priority values to assign.
        """
        if not isinstance(priorities, np.ndarray):
            priorities = np.array(priorities)

        for idx, priority in zip(tree_indices, priorities):
            self.tree.update(idx, priority )

    def __len__(self):
        """Return the current number of items in the buffer."""
        return self.tree.n_entries
