import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from algo.utils import TanhTransform, DoubleQFunc
from algo.utils import Policy, Classifier
from algo.liberty_models import MetricModel, InverseDynamicsModel, ForwardDynamicsModel


class NOMAD_IDBM_RETRIEVAL_FILTERED:
    """
    NOMAD with Inverse Dynamics Bisimulation Metric, Retrieval-Based Cross-Domain Learning,
    and Source Sample Filtering
    
    Key Features:
    1. Separate critics for exploration (intrinsic + extrinsic) and task (extrinsic only)
    2. Dual policies: main policy (source) and target exploration policy
    3. DARC importance sampling for source data
    4. LIBERTY exploration via metric, inverse dynamics, forward dynamics models
    5. **Retrieval-based metric learning**: For each target state, find nearest source state
       and train metric on semantically matched pairs instead of random pairs
    6. **Diversity-seeking exploration**: Intrinsic rewards encourage visiting states DIFFERENT from source
    7. **Source filtering**: Filter out source samples too different from target to prevent policy bias
    
    Mathematical Foundation:
    - Bisimulation metric with inverse dynamics:
      d_inv(s_i, s_j) = |r(s_i) - r(s_j)| 
                       + β·W_2(P(·|s_i), P(·|s_j))
                       + λ·||I(·|s_i,s'_i) - I(·|s_j,s'_j)||_1
    
    - Diversity-seeking potential: Φ(s) = +d_metric(s, s_ref) (positive distance)
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
        self.policy = Policy(config['state_dim'], config['action_dim'], config['hidden_sizes'], 
                            TanhTransform(cache_size=1), config['state_dependent_std']).to(self.device)

        # Target exploration policy (for target environment)
        self.target_policy = Policy(config['state_dim'], config['action_dim'], config['hidden_sizes'], 
                                   TanhTransform(cache_size=1), config['state_dependent_std']).to(self.device)

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
        self.inverse_model = InverseDynamicsModel(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.forward_model = ForwardDynamicsModel(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.intrinsic_reward_coef = config.get('intrinsic_reward_coef', 0.003)

        # Retrieval configuration
        self.retrieval_method = config.get('retrieval_method', 'l2')  # 'l2', 'metric', 'embedding'
        self.retrieval_k = config.get('retrieval_k', 1)  # Number of nearest neighbors
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
        self.dynamics_optimizer = torch.optim.Adam(
            list(self.inverse_model.parameters()) + list(self.forward_model.parameters()),
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

    def retrieve_nearest_source_states(self, target_states, method='l2'):
        """
        For each target state, find the nearest source state from source replay buffer.
        
        Args:
            target_states: (batch_size, state_dim) tensor of target states
            method: 'l2' (raw state distance), 'metric' (learned metric)
        
        Returns:
            matched_src_states: (batch_size, state_dim) tensor of matched source states
            matched_src_next_states: (batch_size, state_dim) tensor of corresponding next states
            matched_src_actions: (batch_size, action_dim) tensor of corresponding actions
        """
        batch_size = target_states.size(0)
        
        # Sample a large batch from source buffer to search within
        search_size = min(self.retrieval_buffer_size, self.src_replay_buffer.size)
        src_state, src_action, src_next_state, _, _ = self.src_replay_buffer.sample(search_size)
        
        if method == 'l2':
            # Compute pairwise L2 distances
            target_norm = (target_states ** 2).sum(dim=1, keepdim=True)  # (batch_size, 1)
            source_norm = (src_state ** 2).sum(dim=1, keepdim=True).t()  # (1, search_size)
            cross_term = torch.mm(target_states, src_state.t())  # (batch_size, search_size)
            distances = target_norm + source_norm - 2 * cross_term  # (batch_size, search_size)
            
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
            distances = distances_flat.reshape(batch_size, search_size).squeeze(-1)  # (batch_size, search_size)
            
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
        
        # Find nearest neighbors
        nearest_indices = distances.argmin(dim=1)  # (batch_size,)
        
        # Retrieve matched source transitions
        matched_src_states = src_state[nearest_indices]  # (batch_size, state_dim)
        matched_src_next_states = src_next_state[nearest_indices]  # (batch_size, state_dim)
        matched_src_actions = src_action[nearest_indices]  # (batch_size, action_dim)
        
        return matched_src_states, matched_src_next_states, matched_src_actions

    def update_classifier(self, src_replay_buffer, tar_replay_buffer, batch_size, writer=None):
        src_state, src_action, src_next_state, _, _ = src_replay_buffer.sample(batch_size)
        tar_state, tar_action, tar_next_state, _, _ = tar_replay_buffer.sample(batch_size)
        state = torch.cat([src_state, tar_state], 0)
        action = torch.cat([src_action, tar_action], 0)
        next_state = torch.cat([src_next_state, tar_next_state], 0)
        label = torch.cat([torch.zeros(size=(batch_size,)), torch.ones(size=(batch_size,))], dim=0).long().to(self.device)

        logit = self.classifier(state, action, next_state)
        classifier_loss = F.cross_entropy(logit, label)
        self.classifier_optimizer.zero_grad()
        classifier_loss.backward()
        self.classifier_optimizer.step()

        if writer is not None and self.total_it % 1000 == 0:
            writer.add_scalar('train/classifier_loss', classifier_loss.mean(), self.total_it)

    def _update_liberty_models(self, tar_replay_buffer, batch_size, writer=None):
        """
        Update LIBERTY exploration models with RETRIEVAL-BASED cross-domain learning.
        """
        # Sample target batch
        tar_state, tar_action, tar_next_state, _, _ = tar_replay_buffer.sample(batch_size)
        
        # RETRIEVAL: Find nearest source states for each target state
        matched_src_state, matched_src_next_state, matched_src_action = \
            self.retrieve_nearest_source_states(tar_state, method=self.retrieval_method)
        
        # 1. METRIC MODEL: Learn bisimulation distance on matched cross-domain pairs
        with torch.no_grad():
            reward_diff = torch.zeros(batch_size, 1).to(self.device)
            
            pred_tar_next = self.forward_model(tar_state, tar_action)
            pred_src_next = self.forward_model(matched_src_state, matched_src_action)
            dynamics_diff = torch.sqrt((pred_tar_next - pred_src_next) ** 2 + 1e-8).mean(dim=1, keepdim=True)
            
            pred_tar_action = self.inverse_model(tar_state, tar_next_state)
            pred_src_action = self.inverse_model(matched_src_state, matched_src_next_state)
            action_diff = torch.abs(pred_tar_action - pred_src_action).mean(dim=1, keepdim=True)
            
            beta = self.config.get('bisim_beta', 0.1)
            lambda_inv = self.config.get('bisim_lambda', 0.1)
            bisim_target = reward_diff + beta * dynamics_diff + lambda_inv * action_diff
        
        pred_distance = self.metric_model(tar_state, matched_src_state)
        metric_loss = F.mse_loss(pred_distance, bisim_target)
        
        # 2. INVERSE DYNAMICS MODEL
        pred_action_tar = self.inverse_model(tar_state, tar_next_state)
        inverse_loss = F.mse_loss(pred_action_tar, tar_action)
        
        # 3. FORWARD DYNAMICS MODEL
        pred_next_state_tar = self.forward_model(tar_state, tar_action)
        forward_loss = F.mse_loss(pred_next_state_tar, tar_next_state)
        
        # Update metric model
        self.metric_optimizer.zero_grad()
        metric_loss.backward()
        self.metric_optimizer.step()
        
        # Update dynamics models
        self.dynamics_optimizer.zero_grad()
        (inverse_loss + forward_loss).backward()
        self.dynamics_optimizer.step()
        
        if writer is not None and self.total_it % 1000 == 0:
            writer.add_scalar('train_liberty/metric_loss', metric_loss.item(), self.total_it)
            writer.add_scalar('train_liberty/inverse_loss', inverse_loss.item(), self.total_it)
            writer.add_scalar('train_liberty/forward_loss', forward_loss.item(), self.total_it)
            writer.add_scalar('train_liberty/bisim_target_mean', bisim_target.mean().item(), self.total_it)

    def train(self, src_replay_buffer, tar_replay_buffer, batch_size=256, writer=None):
        """
        Training with THREE key innovations:
        1. Exploration critic + target policy (intrinsic diversity-seeking + extrinsic rewards)
        2. Task critic + main policy (extrinsic only, with IS correction + source filtering)
        3. LIBERTY models (metric, inverse, forward dynamics) with retrieval-based learning
        """
        self.total_it += 1

        # Update LIBERTY models with RETRIEVAL-BASED cross-domain learning
        self._update_liberty_models(tar_replay_buffer, batch_size, writer)

        # ========== EXPLORATION TRAINING (Target Policy + Exploration Critic) ==========
        tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Compute DIVERSITY-SEEKING intrinsic reward using TRANSITION-BASED potential
            # Retrieve coherent source transition (s_src, a_src, s'_src) for current target state
            matched_src_state, matched_src_next_state, matched_src_action = \
                self.retrieve_nearest_source_states(tar_state, method=self.retrieval_method)
            
            # For next target state, retrieve its matched source transition
            matched_src_state_next, matched_src_next_state_next, matched_src_action_next = \
                self.retrieve_nearest_source_states(tar_next_state, method=self.retrieval_method)
            
            # POTENTIAL FUNCTION: Based on inverse dynamics prediction difference
            # Φ(s, s') = ||I(s_tar, s'_tar) - I(s_src, s'_src)||
            # This captures behavioral difference in state transitions
            
            # Predict actions using inverse dynamics model
            pred_action_tar_current = self.inverse_model(tar_state, tar_next_state)
            pred_action_src_current = self.inverse_model(matched_src_state, matched_src_next_state)
            
            # We need next transition for potential shaping: (s'_tar, s''_tar)
            # But we only have s'_tar, so we use matched source as reference
            pred_action_tar_next = self.inverse_model(tar_next_state, matched_src_state_next)
            pred_action_src_next = self.inverse_model(matched_src_state_next, matched_src_next_state_next)
            
            # Current potential: action prediction difference for current transition
            current_potential_inv = torch.abs(pred_action_tar_current - pred_action_src_current).mean(dim=1, keepdim=True)
            
            # Next potential: action prediction difference for next transition  
            next_potential_inv = torch.abs(pred_action_tar_next - pred_action_src_next).mean(dim=1, keepdim=True)
            
            # Also include state-level distance (from metric model)
            current_potential_state = self.metric_model(tar_state, matched_src_state)
            next_potential_state = self.metric_model(tar_next_state, matched_src_state_next)
            
            # Combined potential: state distance + inverse dynamics difference
            alpha_state = 0.5  # Weight for state distance
            alpha_inv = 0.5    # Weight for inverse dynamics
            
            current_potential = alpha_state * current_potential_state + alpha_inv * current_potential_inv
            next_potential = alpha_state * next_potential_state + alpha_inv * next_potential_inv
            
            # Intrinsic reward: reward for transitions with DIFFERENT behavior from source
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
            # DARC importance sampling weights
            src_logit = self.classifier(src_state, src_action, src_next_state)
            src_prob = F.softmax(src_logit, dim=-1)
            IS_weight_darc = src_prob[:, 1:2] / src_prob[:, 0:1]
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
            _, log_prob_task, _ = self.policy(tar_state_task, tar_action_task, get_logprob=True)
            _, log_prob_exploration, _ = self.target_policy(tar_state_task, tar_action_task, get_logprob=True)
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
        if self.total_it % self.config.get('policy_freq', 2) == 0:
            for p in self.task_q_funcs.parameters(): p.requires_grad = False
            pi_action_src, log_prob_pi_src, _ = self.policy(src_state, get_logprob=True)
            q1_pi_src, q2_pi_src = self.task_q_funcs(src_state, pi_action_src)
            min_q_pi_src = torch.min(q1_pi_src, q2_pi_src)
            policy_loss_src = (self.alpha.detach() * log_prob_pi_src - min_q_pi_src).mean()

            self.policy_optimizer.zero_grad()
            policy_loss_src.backward()
            self.policy_optimizer.step()
            for p in self.task_q_funcs.parameters(): p.requires_grad = True

            # Temperature update
            alpha_loss = -self.log_alpha * (log_prob_pi_src.detach() + self.target_entropy).mean()
            self.temp_optimizer.zero_grad()
            alpha_loss.backward()
            self.temp_optimizer.step()
            self.alpha = self.log_alpha.exp()

            if writer is not None:
                writer.add_scalar('loss/policy_loss', policy_loss_src, self.total_it)
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
            'inverse_model': self.inverse_model.state_dict(),
            'forward_model': self.forward_model.state_dict()
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.policy.load_state_dict(checkpoint['policy'])
        self.target_policy.load_state_dict(checkpoint['target_policy'])
        self.exploration_q_funcs.load_state_dict(checkpoint['exploration_q_funcs'])
        self.task_q_funcs.load_state_dict(checkpoint['task_q_funcs'])
        self.classifier.load_state_dict(checkpoint['classifier'])
        self.metric_model.load_state_dict(checkpoint['metric_model'])
        self.inverse_model.load_state_dict(checkpoint['inverse_model'])
        self.forward_model.load_state_dict(checkpoint['forward_model'])
