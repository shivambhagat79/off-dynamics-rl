import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, constraints
from torch.distributions.transforms import Transform

# Import the new models for LIBERTY exploration
from algo.liberty_models import MetricModel, InverseDynamicsModel, ForwardDynamicsModel


class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


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


class Policy(nn.Module):

    def __init__(self, state_dim, action_dim, max_action, hidden_size=256):
        super(Policy, self).__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.network = MLPNetwork(state_dim, action_dim * 2, hidden_size)

    def forward(self, x, get_logprob=False):
        mu_logstd = self.network(x)
        mu, logstd = mu_logstd.chunk(2, dim=1)
        logstd = torch.clamp(logstd, -20, 2)
        std = logstd.exp()
        dist = Normal(mu, std)
        transforms = [TanhTransform(cache_size=1)]
        dist = TransformedDistribution(dist, transforms)
        action = dist.rsample()
        if get_logprob:
            logprob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        else:
            logprob = None
        mean = torch.tanh(mu)

        return action * self.max_action, logprob, mean * self.max_action

    def log_prob(self, state, action):
        mu_logstd = self.network(state)
        mu, logstd = mu_logstd.chunk(2, dim=1)
        logstd = torch.clamp(logstd, -20, 2)
        std = logstd.exp()
        dist = Normal(mu, std)
        # The action from the buffer is already tanh-squashed and scaled.
        # We need to inverse the transformations to compute the log_prob.
        # Inverse scaling:
        action_unscaled = action / self.max_action
        # Inverse tanh:
        action_untransformed = TanhTransform.atanh(action_unscaled.clamp(-0.99999, 0.99999))

        # Compute log_prob in the untransformed space
        log_prob = dist.log_prob(action_untransformed)

        # Add the correction for the tanh transformation's change of variables
        log_prob -= TanhTransform().log_abs_det_jacobian(action_untransformed, action_unscaled)

        return log_prob.sum(axis=-1, keepdim=True)


class DoubleQFunc(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DoubleQFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)


# domain classifier for DARC
class Classifier(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_size=256, gaussian_noise_std=1.0):
        super(Classifier, self).__init__()
        self.action_dim = action_dim
        self.gaussian_noise_std = gaussian_noise_std
        self.sa_classifier = MLPNetwork(state_dim + action_dim, 2, hidden_size)
        self.sas_classifier = MLPNetwork(2 * state_dim + action_dim, 2, hidden_size)

    def forward(self, state_batch, action_batch, nextstate_batch, with_noise):
        sas = torch.cat([state_batch, action_batch, nextstate_batch], -1)

        if with_noise:
            sas += torch.randn_like(sas, device=state_batch.device) * self.gaussian_noise_std
        sas_logits = F.softmax(self.sas_classifier(sas), dim=1)

        sa = torch.cat([state_batch, action_batch], -1)

        if with_noise:
            sa += torch.randn_like(sa, device=state_batch.device) * self.gaussian_noise_std
        sa_logits = F.softmax(self.sa_classifier(sa), dim=1)

        return sas_logits, sa_logits


class NOMAD_IDBM_RETRIEVAL_KNN:
    """
    NOMAD with Inverse Dynamics Bisimulation Metric, k-NN Retrieval-Based Cross-Domain Learning,
    and Source Sample Filtering
    
    Key Features:
    1. Separate critics for exploration (intrinsic + extrinsic) and task (extrinsic only)
    2. Dual policies: main policy (source) and target exploration policy
    3. DARC importance sampling for source data
    4. LIBERTY exploration via metric, inverse dynamics, forward dynamics models
    5. **k-NN Retrieval-based metric learning**: For each target state, find k nearest source states
       and average potentials for robust intrinsic reward computation
    6. **Diversity-seeking exploration**: Intrinsic rewards encourage visiting states DIFFERENT from source
    7. **Source filtering**: Filter out source samples too different from target to prevent policy bias
    
    Mathematical Foundation:
    - Bisimulation metric with inverse dynamics:
      d_inv(s_i, s_j) = |r(s_i) - r(s_j)| 
                       + β·W_2(P(·|s_i), P(·|s_j))
                       + λ·||I(·|s_i,s'_i) - I(·|s_j,s'_j)||_1
    
    - Diversity-seeking potential (k-NN averaged): Φ(s) = (1/k) Σ_i d_metric(s, s_ref^i)
    - Intrinsic reward: r_intrinsic = γ·Φ(s') - Φ(s)
    
    - Source filtering: Keep only source samples where d_metric(s_src, nearest_target) < threshold
    """
    
    def __init__(self, config, src_replay_buffer, tar_replay_buffer):
        self.config = config
        self.device = torch.device(config['device'])
        self.discount = config['discount']
        self.tau = config['tau']
        self.total_it = 0

        # Exploration mode flag
        self.exploration_mode = False  # Default is source policy
        self.src_replay_buffer = src_replay_buffer
        self.tar_replay_buffer = tar_replay_buffer

        # Temperature parameter
        self.target_entropy = -config['action_dim']
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()

        # Main policy (for source environment)
        self.policy = Policy(config['state_dim'], config['action_dim'], config['max_action'],
                            hidden_size=config['hidden_sizes']).to(self.device)

        # Target exploration policy (for target environment)
        self.target_policy = Policy(config['state_dim'], config['action_dim'], config['max_action'],
                                   hidden_size=config['hidden_sizes']).to(self.device)

        # SEPARATE CRITICS FOR EXPLORATION AND TASK
        # Exploration Q-functions: trained with intrinsic + extrinsic rewards
        self.exploration_q_funcs = DoubleQFunc(config['state_dim'], config['action_dim'], config['hidden_sizes']).to(self.device)
        self.target_exploration_q_funcs = DoubleQFunc(config['state_dim'], config['action_dim'], config['hidden_sizes']).to(self.device)
        self.target_exploration_q_funcs.load_state_dict(self.exploration_q_funcs.state_dict())
        
        # Task Q-functions: trained with extrinsic rewards only (for final evaluation)
        self.task_q_funcs = DoubleQFunc(config['state_dim'], config['action_dim'], config['hidden_sizes']).to(self.device)
        self.target_task_q_funcs = DoubleQFunc(config['state_dim'], config['action_dim'], config['hidden_sizes']).to(self.device)
        self.target_task_q_funcs.load_state_dict(self.task_q_funcs.state_dict())

        # Copy parameters for target policy
        self.target_policy.load_state_dict(self.policy.state_dict())
        
        # DARC domain classifier
        self.classifier = Classifier(config['state_dim'], config['action_dim'], config['hidden_sizes'], config['gaussian_noise_std']).to(self.device)

        # LIBERTY Exploration Components
        self.metric_model = MetricModel(config['state_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        
        # SEPARATE dynamics models for source and target domains (different physics)
        self.src_inverse_model = InverseDynamicsModel(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.src_forward_model = ForwardDynamicsModel(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.tar_inverse_model = InverseDynamicsModel(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.tar_forward_model = ForwardDynamicsModel(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        
        self.intrinsic_reward_coef = config.get('intrinsic_reward_coef', 0.003)

        # Retrieval configuration
        self.retrieval_method = config.get('retrieval_method', 'l2')  # 'l2', 'metric'
        self.retrieval_k = config.get('retrieval_k', 5)  # Number of nearest neighbors for averaging
        self.retrieval_buffer_size = config.get('retrieval_buffer_size', 10000)  # Max source states to search
        
        # Source filtering configuration
        self.use_source_filtering = config.get('use_source_filtering', True)  # Whether to filter source samples
        self.filter_start_step = config.get('filter_start_step', 5000)  # Start filtering after this many steps
        self.filter_threshold_quantile = config.get('filter_threshold_quantile', 0.7)  # Keep top 70% most similar samples
        self.filter_method = config.get('filter_method', 'metric')  # 'metric' or 'classifier'
        
        # Optimizers
        self.exploration_q_optimizer = torch.optim.Adam(self.exploration_q_funcs.parameters(), lr=config['critic_lr'])
        self.task_q_optimizer = torch.optim.Adam(self.task_q_funcs.parameters(), lr=config['critic_lr'])
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['actor_lr'])
        self.target_policy_optimizer = torch.optim.Adam(self.target_policy.parameters(), lr=config.get('exploration_actor_lr', config['actor_lr']))
        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=config['actor_lr'])
        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=config['actor_lr'])
        self.metric_optimizer = torch.optim.Adam(self.metric_model.parameters(), lr=config.get('liberty_lr', 3e-4))
        self.src_dynamics_optimizer = torch.optim.Adam(
            list(self.src_inverse_model.parameters()) + list(self.src_forward_model.parameters()),
            lr=config.get('liberty_lr', 3e-4)
        )
        self.tar_dynamics_optimizer = torch.optim.Adam(
            list(self.tar_inverse_model.parameters()) + list(self.tar_forward_model.parameters()),
            lr=config.get('liberty_lr', 3e-4)
        )

    def select_action(self, state, test=True):
        with torch.no_grad():
            if self.exploration_mode and not test:
                # Use target policy for collecting experience in the target environment
                policy_to_use = self.target_policy
            else:
                # Use the main source policy for source env and for all testing
                policy_to_use = self.policy
            action, _, mean = policy_to_use(torch.Tensor(state).view(1,-1).to(self.device))
        if test:
            return mean.squeeze().cpu().numpy()
        else:
            return action.squeeze().cpu().numpy()

    def enable_exploration(self):
        """Switches to the target policy for action selection during training."""
        self.exploration_mode = True

    def disable_exploration(self):
        """Switches back to the main source policy for action selection."""
        self.exploration_mode = False

    def retrieve_nearest_target_states(self, source_states, method='l2'):
        """
        For each source state, find the nearest target state from target replay buffer.
        Used for filtering source samples.
        
        Args:
            source_states: (batch_size, state_dim) tensor of source states
            method: 'l2' (raw state distance), 'metric' (learned metric)
        
        Returns:
            matched_tar_states: (batch_size, state_dim) tensor of matched target states
            distances: (batch_size,) tensor of distances to nearest target states
        """
        batch_size = source_states.size(0)
        
        # Sample target buffer to search within
        search_size = min(self.retrieval_buffer_size, self.tar_replay_buffer.size)
        tar_state, _, _, _, _ = self.tar_replay_buffer.sample(search_size)
        
        if method == 'l2':
            # Compute pairwise L2 distances
            source_norm = (source_states ** 2).sum(dim=1, keepdim=True)  # (batch_size, 1)
            target_norm = (tar_state ** 2).sum(dim=1, keepdim=True).t()  # (1, search_size)
            cross_term = torch.mm(source_states, tar_state.t())  # (batch_size, search_size)
            distances_matrix = source_norm + target_norm - 2 * cross_term  # (batch_size, search_size)
            
        elif method == 'metric':
            # Use learned metric model
            source_expanded = source_states.unsqueeze(1)  # (batch_size, 1, state_dim)
            target_expanded = tar_state.unsqueeze(0)  # (1, search_size, state_dim)
            
            source_repeated = source_expanded.repeat(1, search_size, 1)  # (batch_size, search_size, state_dim)
            target_repeated = target_expanded.repeat(batch_size, 1, 1)  # (batch_size, search_size, state_dim)
            
            source_flat = source_repeated.reshape(-1, source_states.size(1))
            target_flat = target_repeated.reshape(-1, tar_state.size(1))
            
            with torch.no_grad():
                distances_flat = self.metric_model(source_flat, target_flat)  # (batch_size * search_size, 1)
            distances_matrix = distances_flat.reshape(batch_size, search_size).squeeze(-1)  # (batch_size, search_size)
            
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
        
        # Find nearest neighbors
        min_distances, nearest_indices = distances_matrix.min(dim=1)  # (batch_size,)
        matched_tar_states = tar_state[nearest_indices]  # (batch_size, state_dim)
        
        return matched_tar_states, min_distances

    def retrieve_knn_source_states(self, target_states, method='l2', k=None):
        """
        For each target state, find the k nearest source states from source replay buffer.
        
        Args:
            target_states: (batch_size, state_dim) tensor of target states
            method: 'l2' (raw state distance), 'metric' (learned metric)
            k: number of nearest neighbors (default: self.retrieval_k)
        
        Returns:
            matched_src_states: (batch_size, k, state_dim) tensor of k matched source states
            matched_src_next_states: (batch_size, k, state_dim) tensor of k corresponding next states
            matched_src_actions: (batch_size, k, action_dim) tensor of k corresponding actions
            matched_src_rewards: (batch_size, k, 1) tensor of k corresponding rewards
            distances: (batch_size, k) tensor of distances to k nearest neighbors
        """
        if k is None:
            k = self.retrieval_k
            
        batch_size = target_states.size(0)
        
        # Sample a large batch from source buffer to search within
        search_size = min(self.retrieval_buffer_size, self.src_replay_buffer.size)
        src_state, src_action, src_next_state, src_reward, _ = self.src_replay_buffer.sample(search_size)
        
        if method == 'l2':
            # Compute pairwise L2 distances
            target_norm = (target_states ** 2).sum(dim=1, keepdim=True)  # (batch_size, 1)
            source_norm = (src_state ** 2).sum(dim=1, keepdim=True).t()  # (1, search_size)
            cross_term = torch.mm(target_states, src_state.t())  # (batch_size, search_size)
            distances_matrix = target_norm + source_norm - 2 * cross_term  # (batch_size, search_size)
            
        elif method == 'metric':
            # Use learned metric model to compute distances
            target_expanded = target_states.unsqueeze(1)  # (batch_size, 1, state_dim)
            source_expanded = src_state.unsqueeze(0)  # (1, search_size, state_dim)
            
            target_repeated = target_expanded.repeat(1, search_size, 1)  # (batch_size, search_size, state_dim)
            source_repeated = source_expanded.repeat(batch_size, 1, 1)  # (batch_size, search_size, state_dim)
            
            target_flat = target_repeated.reshape(-1, target_states.size(1))
            source_flat = source_repeated.reshape(-1, src_state.size(1))
            
            with torch.no_grad():
                distances_flat = self.metric_model(target_flat, source_flat)  # (batch_size * search_size, 1)
            distances_matrix = distances_flat.reshape(batch_size, search_size).squeeze(-1)  # (batch_size, search_size)
            
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
        
        # Find k nearest neighbors
        k_actual = min(k, search_size)  # Handle case where k > search_size
        distances_sorted, nearest_indices = distances_matrix.topk(k_actual, dim=1, largest=False)  # (batch_size, k)
        
        # Retrieve matched source transitions for all k neighbors
        matched_src_states = src_state[nearest_indices]  # (batch_size, k, state_dim)
        matched_src_next_states = src_next_state[nearest_indices]  # (batch_size, k, state_dim)
        matched_src_actions = src_action[nearest_indices]  # (batch_size, k, action_dim)
        matched_src_rewards = src_reward[nearest_indices]  # (batch_size, k, 1)
        
        return matched_src_states, matched_src_next_states, matched_src_actions, matched_src_rewards, distances_sorted

    def update_classifier(self, src_replay_buffer, tar_replay_buffer, batch_size, writer=None):
        src_state, src_action, src_next_state, _, _ = src_replay_buffer.sample(batch_size)
        tar_state, tar_action, tar_next_state, _, _ = tar_replay_buffer.sample(batch_size)
        state = torch.cat([src_state, tar_state], 0)
        action = torch.cat([src_action, tar_action], 0)
        next_state = torch.cat([src_next_state, tar_next_state], 0)
        label = torch.cat([torch.zeros(size=(batch_size,)), torch.ones(size=(batch_size,))], dim=0).long().to(self.device)

        sas_logits, sa_logits = self.classifier(state, action, next_state, with_noise=True)
        classifier_loss = F.cross_entropy(sas_logits, label) + F.cross_entropy(sa_logits, label)
        self.classifier_optimizer.zero_grad()
        classifier_loss.backward()
        self.classifier_optimizer.step()

        if writer is not None and self.total_it % 1000 == 0:
            writer.add_scalar('train/classifier_loss', classifier_loss.mean(), self.total_it)
    def _update_liberty_models(self, tar_replay_buffer, batch_size, writer=None):
        """
        Update LIBERTY exploration models with k-NN RETRIEVAL-BASED cross-domain learning.
        For metric training, we use the NEAREST neighbor (k=1) to maintain consistency.
        """
        # Sample target batch WITH rewards
        tar_state, tar_action, tar_next_state, tar_reward, _ = tar_replay_buffer.sample(batch_size)
        
        # RETRIEVAL: Find nearest source states (k=1 for metric training) WITH rewards
        # IMPORTANT: Always use 'l2' for metric training to avoid circular dependency
        # (metric cannot use itself to find training pairs)
        matched_src_states, matched_src_next_states, matched_src_actions, matched_src_rewards, _ = \
            self.retrieve_knn_source_states(tar_state, method='l2', k=1)
        
        # Squeeze k dimension since k=1
        matched_src_state = matched_src_states.squeeze(1)  # (batch_size, state_dim)
        matched_src_next_state = matched_src_next_states.squeeze(1)  # (batch_size, state_dim)
        matched_src_action = matched_src_actions.squeeze(1)  # (batch_size, action_dim)
        matched_src_reward = matched_src_rewards.squeeze(1)  # (batch_size, 1)
        
        # 1. TRAIN TARGET DYNAMICS MODELS (on target data)
        pred_action_tar = self.tar_inverse_model(tar_state, tar_next_state)
        tar_inverse_loss = F.mse_loss(pred_action_tar, tar_action)
        
        pred_next_state_tar_mu, _ = self.tar_forward_model(tar_state, tar_action)
        tar_forward_loss = F.mse_loss(pred_next_state_tar_mu, tar_next_state)
        
        self.tar_dynamics_optimizer.zero_grad()
        (tar_inverse_loss + tar_forward_loss).backward()
        self.tar_dynamics_optimizer.step()
        
        # 2. TRAIN SOURCE DYNAMICS MODELS (on FULL source distribution, not just retrieval matches)
        # Sample independently from source buffer to avoid biased training distribution
        src_state_full, src_action_full, src_next_state_full, _, _ = self.src_replay_buffer.sample(batch_size)
        
        pred_action_src = self.src_inverse_model(src_state_full, src_next_state_full)
        src_inverse_loss = F.mse_loss(pred_action_src, src_action_full)
        
        pred_next_state_src_mu, _ = self.src_forward_model(src_state_full, src_action_full)
        src_forward_loss = F.mse_loss(pred_next_state_src_mu, src_next_state_full)
        
        self.src_dynamics_optimizer.zero_grad()
        (src_inverse_loss + src_forward_loss).backward()
        self.src_dynamics_optimizer.step()
        
        # 3. TRAIN METRIC MODEL: Learn cross-domain distance using DOMAIN-SPECIFIC models
        with torch.no_grad():
            # Actual reward difference across domains
            reward_diff = torch.abs(tar_reward - matched_src_reward)
            
            # Use TARGET model for target states, SOURCE model for source states
            pred_tar_next_mu, _ = self.tar_forward_model(tar_state, tar_action)
            pred_src_next_mu, _ = self.src_forward_model(matched_src_state, matched_src_action)
            dynamics_diff = torch.sqrt((pred_tar_next_mu - pred_src_next_mu) ** 2 + 1e-8).mean(dim=1, keepdim=True)
            
            pred_tar_action = self.tar_inverse_model(tar_state, tar_next_state)
            pred_src_action = self.src_inverse_model(matched_src_state, matched_src_next_state)
            action_diff = torch.abs(pred_tar_action - pred_src_action).mean(dim=1, keepdim=True)
            
            beta = self.config.get('bisim_beta', 0.1)
            lambda_inv = self.config.get('bisim_lambda', 0.1)
            bisim_target = reward_diff + beta * dynamics_diff + lambda_inv * action_diff
        
        pred_distance = self.metric_model(tar_state, matched_src_state)
        metric_loss = F.mse_loss(pred_distance, bisim_target)
        
        self.metric_optimizer.zero_grad()
        metric_loss.backward()
        self.metric_optimizer.step()
        
        if writer is not None and self.total_it % 1000 == 0:
            writer.add_scalar('train_liberty/metric_loss', metric_loss.item(), self.total_it)
            writer.add_scalar('train_liberty/tar_inverse_loss', tar_inverse_loss.item(), self.total_it)
            writer.add_scalar('train_liberty/tar_forward_loss', tar_forward_loss.item(), self.total_it)
            writer.add_scalar('train_liberty/src_inverse_loss', src_inverse_loss.item(), self.total_it)
            writer.add_scalar('train_liberty/src_forward_loss', src_forward_loss.item(), self.total_it)
            writer.add_scalar('train_liberty/bisim_target_mean', bisim_target.mean().item(), self.total_it)

    def compute_knn_potential(self, target_states, matched_src_states):
        """
        Compute potential function by averaging over k nearest neighbors.
        
        Args:
            target_states: (batch_size, state_dim)
            matched_src_states: (batch_size, k, state_dim)
        
        Returns:
            potential: (batch_size, 1) - averaged potential over k neighbors
        """
        batch_size = target_states.size(0)
        k = matched_src_states.size(1)
        
        # Expand target states to match k dimension
        target_expanded = target_states.unsqueeze(1).expand(-1, k, -1)  # (batch_size, k, state_dim)
        
        # Flatten for batch processing
        target_flat = target_expanded.reshape(batch_size * k, -1)  # (batch_size * k, state_dim)
        source_flat = matched_src_states.reshape(batch_size * k, -1)  # (batch_size * k, state_dim)
        
        # Compute potentials for all k neighbors
        with torch.no_grad():
            potentials_flat = self.metric_model(target_flat, source_flat)  # (batch_size * k, 1)
        
        # Reshape and average over k neighbors
        potentials = potentials_flat.reshape(batch_size, k)  # (batch_size, k)
        avg_potential = potentials.mean(dim=1, keepdim=True)  # (batch_size, 1)
        
        return avg_potential

    def train(self, src_replay_buffer, tar_replay_buffer, batch_size=256, writer=None):
        """
        Training with THREE key innovations:
        1. Exploration critic + target policy (k-NN averaged intrinsic diversity-seeking + extrinsic rewards)
        2. Task critic + main policy (extrinsic only, with IS correction + source filtering)
        3. LIBERTY models (metric, inverse, forward dynamics) with retrieval-based learning
        """
        self.total_it += 1

        # Update domain classifier for DARC importance sampling (with warmup and reduced frequency)
        if self.total_it > self.config.get('darc_warmup_steps', 10000):
            if self.total_it % self.config.get('classifier_update_freq', 10) == 0:
                self.update_classifier(src_replay_buffer, tar_replay_buffer, batch_size, writer)

        # Update LIBERTY models with RETRIEVAL-BASED cross-domain learning
        self._update_liberty_models(tar_replay_buffer, batch_size, writer)

        # ========== EXPLORATION TRAINING (Target Policy + Exploration Critic) ==========
        tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Compute DIVERSITY-SEEKING intrinsic reward using k-NN averaging
            matched_src_states, _, _, _, _ = self.retrieve_knn_source_states(
                tar_state, method=self.retrieval_method, k=self.retrieval_k
            )
            matched_src_next_states, _, _, _, _ = self.retrieve_knn_source_states(
                tar_next_state, method=self.retrieval_method, k=self.retrieval_k
            )
            
            # Compute averaged potential over k neighbors for robustness
            current_potential = self.compute_knn_potential(tar_state, matched_src_states)
            next_potential = self.compute_knn_potential(tar_next_state, matched_src_next_states)
            
            # Positive distance = reward for being DIFFERENT from source (averaged over k neighbors)
            intrinsic_reward = self.discount * next_potential - current_potential
            
            tar_reward += self.intrinsic_reward_coef * intrinsic_reward

            next_action_tar, log_prob_tar, _ = self.target_policy(tar_next_state, get_logprob=True)
            q_t1, q_t2 = self.target_exploration_q_funcs(tar_next_state, next_action_tar)
            q_target = torch.min(q_t1, q_t2)
            value_target_tar = tar_reward + tar_not_done * self.discount * (q_target - self.alpha.detach() * log_prob_tar)

        # Update EXPLORATION critic
        q1, q2 = self.exploration_q_funcs(tar_state, tar_action)
        q_loss_exploration = F.mse_loss(q1, value_target_tar) + F.mse_loss(q2, value_target_tar)

        self.exploration_q_optimizer.zero_grad()
        q_loss_exploration.backward()
        self.exploration_q_optimizer.step()

        # Update exploratory policy
        if self.total_it % self.config.get('policy_freq', 2) == 0:
            for p in self.exploration_q_funcs.parameters(): p.requires_grad = False
            pi_action_tar, log_prob_pi_tar, _ = self.target_policy(tar_state, get_logprob=True)
            q1_pi, q2_pi = self.exploration_q_funcs(tar_state, pi_action_tar)
            min_q_pi = torch.min(q1_pi, q2_pi)
            policy_loss_tar = (self.alpha.detach() * log_prob_pi_tar - min_q_pi).mean()

            self.target_policy_optimizer.zero_grad()
            policy_loss_tar.backward()
            self.target_policy_optimizer.step()
            for p in self.exploration_q_funcs.parameters(): p.requires_grad = True

            if writer is not None:
                writer.add_scalar('loss/target_policy_loss', policy_loss_tar, self.total_it)

        # ========== TASK TRAINING (Main Policy + Task Critic) ==========
        src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample(batch_size)

        with torch.no_grad():
            # DARC importance sampling weights: ρ(s,a,s') = P_SAS(tar|sas) * P_SA(src|sa) / (P_SA(tar|sa) * P_SAS(src|sas))
            sas_logits_src, sa_logits_src = self.classifier(src_state, src_action, src_next_state, with_noise=False)
            sas_probs_src = F.softmax(sas_logits_src, dim=-1)
            sa_probs_src = F.softmax(sa_logits_src, dim=-1)
            darc_log_ratio_src = torch.log(sas_probs_src[:, 1:] + 1e-8) + torch.log(sa_probs_src[:, 0:1] + 1e-8) - \
                                 torch.log(sa_probs_src[:, 1:] + 1e-8) - torch.log(sas_probs_src[:, 0:1] + 1e-8)
            IS_weight_darc = torch.exp(darc_log_ratio_src).detach()
            IS_weight_darc = torch.clamp(IS_weight_darc, 1e-4, 1.0)
            
            # ========== SOURCE FILTERING ==========
            # Filter out source samples too different from target domain
            if self.use_source_filtering and self.total_it > self.filter_start_step:
                if self.filter_method == 'metric':
                    # Find nearest target state for each source state
                    _, distances_to_target = self.retrieve_nearest_target_states(src_state, method='metric')
                    
                    # Dynamic threshold: keep top X% most similar samples
                    threshold_distance = torch.quantile(distances_to_target, self.filter_threshold_quantile)
                    
                    # Filter mask: 1.0 for accepted, 0.0 for rejected
                    filter_mask = (distances_to_target <= threshold_distance).float().unsqueeze(1)
                    
                elif self.filter_method == 'classifier':
                    # Use classifier probability as filter
                    classifier_prob_target = src_prob[:, 1:2]
                    threshold_prob = torch.quantile(classifier_prob_target.squeeze(), 1.0 - self.filter_threshold_quantile)
                    filter_mask = (classifier_prob_target >= threshold_prob).float()
                else:
                    filter_mask = torch.ones_like(src_reward)
            else:
                # No filtering initially (metric needs to be trained first)
                filter_mask = torch.ones_like(src_reward)
            
            # Combine filtering with importance sampling
            IS_weight_darc = IS_weight_darc * filter_mask

        # Sample target data
        tar_state_task, tar_action_task, tar_next_state_task, tar_reward_task, tar_not_done_task = tar_replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Policy importance sampling
            log_prob_task = self.policy.log_prob(tar_state_task, tar_action_task)
            log_prob_exploration = self.target_policy.log_prob(tar_state_task, tar_action_task)
            IS_weight_policy = torch.exp(log_prob_task - log_prob_exploration)
            IS_weight_policy = torch.clamp(IS_weight_policy, 1e-4, 1.0)

            # Compute target Q values (NO intrinsic reward for task critic)
            next_action_src, log_prob_src, _ = self.policy(src_next_state, get_logprob=True)
            q_t1_src, q_t2_src = self.target_task_q_funcs(src_next_state, next_action_src)
            q_target_src = torch.min(q_t1_src, q_t2_src)
            value_target_src = src_reward + src_not_done * self.discount * (q_target_src - self.alpha.detach() * log_prob_src)

            next_action_tar_task, log_prob_tar_task, _ = self.policy(tar_next_state_task, get_logprob=True)
            q_t1_tar, q_t2_tar = self.target_task_q_funcs(tar_next_state_task, next_action_tar_task)
            q_target_tar = torch.min(q_t1_tar, q_t2_tar)
            value_target_tar_task = tar_reward_task + tar_not_done_task * self.discount * (q_target_tar - self.alpha.detach() * log_prob_tar_task)

        # Update TASK critic with filtered + importance-weighted data
        q1_src, q2_src = self.task_q_funcs(src_state, src_action)
        q1_tar_task, q2_tar_task = self.task_q_funcs(tar_state_task, tar_action_task)

        q_loss_task = (IS_weight_darc * (F.mse_loss(q1_src, value_target_src, reduction='none') + 
                                         F.mse_loss(q2_src, value_target_src, reduction='none'))).mean() + \
                      (IS_weight_policy * (F.mse_loss(q1_tar_task, value_target_tar_task, reduction='none') + 
                                          F.mse_loss(q2_tar_task, value_target_tar_task, reduction='none'))).mean()

        self.task_q_optimizer.zero_grad()
        q_loss_task.backward()
        self.task_q_optimizer.step()

        # Update main policy using TASK critic
        # Following Eq. (taskpoliyloss): s ~ B_src ∪ B_tar
        if self.total_it % self.config.get('policy_freq', 2) == 0:
            for p in self.task_q_funcs.parameters(): p.requires_grad = False
            
            # Concatenate source and target states for policy update
            states_combined = torch.cat([src_state, tar_state_task], dim=0)
            
            # Policy gradient on combined batch
            pi_action_combined, log_prob_pi_combined, _ = self.policy(states_combined, get_logprob=True)
            q1_pi_combined, q2_pi_combined = self.task_q_funcs(states_combined, pi_action_combined)
            min_q_pi_combined = torch.min(q1_pi_combined, q2_pi_combined)
            policy_loss = (self.alpha.detach() * log_prob_pi_combined - min_q_pi_combined).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            for p in self.task_q_funcs.parameters(): p.requires_grad = True

            # Temperature update (on combined batch)
            alpha_loss = -self.log_alpha * (log_prob_pi_combined.detach() + self.target_entropy).mean()
            self.temp_optimizer.zero_grad()
            alpha_loss.backward()
            self.temp_optimizer.step()
            self.alpha = self.log_alpha.exp()

            if writer is not None:
                writer.add_scalar('loss/policy_loss', policy_loss, self.total_it)
                writer.add_scalar('loss/alpha_loss', alpha_loss, self.total_it)
                writer.add_scalar('loss/alpha', self.alpha, self.total_it)

        # Update target networks
        if self.total_it % self.config.get('target_update_freq', 2) == 0:
            for param, target_param in zip(self.exploration_q_funcs.parameters(), self.target_exploration_q_funcs.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.task_q_funcs.parameters(), self.target_task_q_funcs.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if writer is not None:
            writer.add_scalar('loss/q_loss_exploration', q_loss_exploration, self.total_it)
            writer.add_scalar('loss/q_loss_task', q_loss_task, self.total_it)
            if self.total_it % 1000 == 0:
                writer.add_scalar('train/intrinsic_reward_mean', intrinsic_reward.mean(), self.total_it)
                writer.add_scalar('train/IS_weight_darc_mean', IS_weight_darc.mean(), self.total_it)
                writer.add_scalar('train/IS_weight_policy_mean', IS_weight_policy.mean(), self.total_it)
                writer.add_scalar('train/retrieval_k', float(self.retrieval_k), self.total_it)
                if self.use_source_filtering and self.total_it > self.filter_start_step:
                    writer.add_scalar('train/source_filter_accept_ratio', filter_mask.mean(), self.total_it)
                    if self.filter_method == 'metric':
                        writer.add_scalar('train/source_target_distance_mean', distances_to_target.mean(), self.total_it)
                        writer.add_scalar('train/source_target_distance_threshold', threshold_distance, self.total_it)

    def save(self, filename):
        torch.save({
            'policy': self.policy.state_dict(),
            'target_policy': self.target_policy.state_dict(),
            'exploration_q_funcs': self.exploration_q_funcs.state_dict(),
            'task_q_funcs': self.task_q_funcs.state_dict(),
            'classifier': self.classifier.state_dict(),
            'metric_model': self.metric_model.state_dict(),
            'src_inverse_model': self.src_inverse_model.state_dict(),
            'src_forward_model': self.src_forward_model.state_dict(),
            'tar_inverse_model': self.tar_inverse_model.state_dict(),
            'tar_forward_model': self.tar_forward_model.state_dict()
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.policy.load_state_dict(checkpoint['policy'])
        self.target_policy.load_state_dict(checkpoint['target_policy'])
        self.exploration_q_funcs.load_state_dict(checkpoint['exploration_q_funcs'])
        self.task_q_funcs.load_state_dict(checkpoint['task_q_funcs'])
        self.classifier.load_state_dict(checkpoint['classifier'])
        self.metric_model.load_state_dict(checkpoint['metric_model'])
        self.src_inverse_model.load_state_dict(checkpoint['src_inverse_model'])
        self.src_forward_model.load_state_dict(checkpoint['src_forward_model'])
        self.tar_inverse_model.load_state_dict(checkpoint['tar_inverse_model'])
        self.tar_forward_model.load_state_dict(checkpoint['tar_forward_model'])
