import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, constraints
from torch.distributions.transforms import Transform


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
        action_unscaled = action / self.max_action
        action_untransformed = TanhTransform.atanh(action_unscaled.clamp(-0.99999, 0.99999))
        log_prob = dist.log_prob(action_untransformed)
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


class Classifier(nn.Module):
    """DARC domain classifier for dynamics importance sampling."""
    
    def __init__(self, state_dim, action_dim, hidden_sizes, gaussian_noise_std=0.5):
        super(Classifier, self).__init__()
        self.gaussian_noise_std = gaussian_noise_std
        
        # s-a-s' classifier (full dynamics)
        self.sas_network = nn.Sequential(
            nn.Linear(state_dim * 2 + action_dim, hidden_sizes),
            nn.ReLU(),
            nn.Linear(hidden_sizes, hidden_sizes),
            nn.ReLU(),
            nn.Linear(hidden_sizes, 2)
        )
        
        # s-a classifier (policy/behavior)
        self.sa_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_sizes),
            nn.ReLU(),
            nn.Linear(hidden_sizes, hidden_sizes),
            nn.ReLU(),
            nn.Linear(hidden_sizes, 2)
        )
    
    def forward(self, state, action, next_state, with_noise=False):
        if with_noise:
            state = state + torch.randn_like(state) * self.gaussian_noise_std
            action = action + torch.randn_like(action) * self.gaussian_noise_std
            next_state = next_state + torch.randn_like(next_state) * self.gaussian_noise_std
        
        sas_input = torch.cat([state, action, next_state], dim=1)
        sas_logits = self.sas_network(sas_input)
        sas_probs = F.softmax(sas_logits, dim=1)
        
        sa_input = torch.cat([state, action], dim=1)
        sa_logits = self.sa_network(sa_input)
        sa_probs = F.softmax(sa_logits, dim=1)
        
        return sas_probs, sa_probs


class CoinFlipNetwork(nn.Module):
    """Neural network that predicts coin flip labels for count-based exploration."""
    
    def __init__(self, state_dim, output_dim=100, hidden_size=256):
        super(CoinFlipNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            # No activation - predict continuous values
        )
    
    def forward(self, state):
        return self.network(state)


class CoinFlipMaker:
    """Generates random coin flip labels for states."""
    
    def __init__(self, output_dim, p_replace=1.0, only_zeros=False):
        self.output_dim = output_dim
        self.p_replace = p_replace
        self.only_zeros = only_zeros
        self.previous_output = self._draw()
    
    def _draw(self):
        if self.only_zeros:
            return np.zeros(self.output_dim, dtype=np.float32)
        # Return -1 or +1
        return 2 * np.random.binomial(1, 0.5, size=self.output_dim).astype(np.float32) - 1
    
    def __call__(self):
        if self.only_zeros:
            return np.zeros(self.output_dim, dtype=np.float32)
        
        new_output = self._draw()
        # With probability p_replace, use new flip; otherwise keep previous
        replace_mask = np.random.rand(self.output_dim) < self.p_replace
        new_output = np.where(replace_mask, new_output, self.previous_output)
        self.previous_output = new_output
        return new_output
    
    def reset(self):
        self.previous_output = self._draw()


class NOMAD_CFN:
    """NOMAD with CFN (Coin Flip Network) count-based exploration."""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.discount = config['gamma']
        self.tau = config['tau']
        self.alpha = torch.tensor(config['alpha'], device=device, requires_grad=False)
        self.log_alpha = torch.log(self.alpha).requires_grad_()
        self.target_entropy = -config['action_dim']
        self.total_it = 0
        self.exploration_mode = False
        
        # Action noise for TD3-style smoothing
        self.temp = config.get('temp', 0.2)
        self.policy_noise_clip = config.get('policy_noise_clip', 0.5)
        
        # CFN parameters (following CFN paper: 20 dims for MuJoCo continuous control)
        self.cfn_output_dim = config.get('cfn_output_dim', 20)
        self.cfn_bonus_exponent = config.get('cfn_bonus_exponent', 0.5)
        self.intrinsic_reward_scale = config.get('intrinsic_reward_scale', 1.0)
        self.cfn_p_replace = config.get('cfn_p_replace', 1.0)
        self.use_reward_normalization = config.get('cfn_reward_normalization', True)
        
        # Running statistics for reward normalization
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 0
        
        # Networks
        self.policy = Policy(config['state_dim'], config['action_dim'], 
                           config['max_action'], config['hidden_sizes']).to(device)
        self.exploration_policy = Policy(config['state_dim'], config['action_dim'], 
                                        config['max_action'], config['hidden_sizes']).to(device)
        
        # Separate critics for task and exploration
        self.task_q_funcs = DoubleQFunc(config['state_dim'], config['action_dim'], 
                                       config['hidden_sizes']).to(device)
        self.target_task_q_funcs = copy.deepcopy(self.task_q_funcs)
        
        self.exploration_q_funcs = DoubleQFunc(config['state_dim'], config['action_dim'], 
                                              config['hidden_sizes']).to(device)
        self.target_exploration_q_funcs = copy.deepcopy(self.exploration_q_funcs)
        
        # DARC domain classifier
        self.classifier = Classifier(config['state_dim'], config['action_dim'], 
                                    config['hidden_sizes'], config['gaussian_noise_std']).to(device)
        
        # CFN components
        self.cfn_network = CoinFlipNetwork(
            config['state_dim'], 
            output_dim=self.cfn_output_dim,
            hidden_size=config['hidden_sizes']
        ).to(device)
        self.coin_flip_maker = CoinFlipMaker(
            self.cfn_output_dim, 
            p_replace=self.cfn_p_replace
        )
        
        # Optimizers
        self.exploration_q_optimizer = torch.optim.Adam(self.exploration_q_funcs.parameters(), lr=config['critic_lr'])
        self.task_q_optimizer = torch.optim.Adam(self.task_q_funcs.parameters(), lr=config['critic_lr'])
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['actor_lr'])
        self.exploration_policy_optimizer = torch.optim.Adam(
            self.exploration_policy.parameters(), 
            lr=config.get('exploration_actor_lr', config['actor_lr'])
        )
        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=config['actor_lr'])
        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=config['actor_lr'])
        self.cfn_optimizer = torch.optim.Adam(
            self.cfn_network.parameters(), 
            lr=config.get('cfn_lr', 3e-4)
        )
    
    def select_action(self, state, test=True):
        with torch.no_grad():
            if self.exploration_mode and not test:
                policy_to_use = self.exploration_policy
            else:
                policy_to_use = self.policy
            action, _, mean = policy_to_use(torch.Tensor(state).view(1,-1).to(self.device))
        if test:
            return mean.squeeze().cpu().numpy()
        else:
            return action.squeeze().cpu().numpy()
    
    def enable_exploration(self):
        self.exploration_mode = True
    
    def disable_exploration(self):
        self.exploration_mode = False
    
    def update_classifier(self, src_replay_buffer, tar_replay_buffer, batch_size, writer=None):
        src_state, src_action, src_next_state, _, _ = src_replay_buffer.sample(batch_size)
        tar_state, tar_action, tar_next_state, _, _ = tar_replay_buffer.sample(batch_size)
        state = torch.cat([src_state, tar_state], 0)
        action = torch.cat([src_action, tar_action], 0)
        next_state = torch.cat([src_next_state, tar_next_state], 0)
        label = torch.cat([torch.zeros(size=(batch_size,)), torch.ones(size=(batch_size,))], 
                         dim=0).long().to(self.device)
        
        indices = torch.randperm(label.shape[0])
        state_batch, action_batch, nextstate_batch, label_batch = \
            state[indices], action[indices], next_state[indices], label[indices]
        
        sas_logits, sa_logits = self.classifier(state_batch, action_batch, nextstate_batch, with_noise=True)
        loss = F.cross_entropy(sas_logits, label_batch) + F.cross_entropy(sa_logits, label_batch)
        
        self.classifier_optimizer.zero_grad()
        loss.backward()
        self.classifier_optimizer.step()
        
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/classifier_loss', loss.item(), global_step=self.total_it)
    

    
    def compute_intrinsic_reward(self, state, update_stats=True):
        """
        Compute CFN intrinsic reward based on prediction magnitude (inverse count estimate).
        
        Following actual CFN implementation:
        one_over_counts = mean(predicted_flips^2)
        magnitude = one_over_counts ^ bonus_exponent
        
        Novel states -> network outputs ~0 -> low magnitude -> low reward (after normalization baseline)
        Familiar states -> network outputs strong predictions -> high magnitude
        When combined with random prior, the variance in predictions serves as novelty signal.
        """
        with torch.no_grad():
            predicted_flips = self.cfn_network(state)
            
            # Compute mean squared prediction (inverse count estimate)
            one_over_counts = torch.mean(predicted_flips ** 2, dim=-1)
            
            # Apply exponent (default 0.5 = sqrt)
            magnitude = one_over_counts ** self.cfn_bonus_exponent
            
            if self.use_reward_normalization and update_stats:
                # Update running statistics
                self.reward_count += 1
                delta = magnitude.item() - self.reward_mean
                self.reward_mean += delta / self.reward_count
                delta2 = magnitude.item() - self.reward_mean
                self.reward_var += delta * delta2
                
                # Normalize (reward = scale * (magnitude - mean) / std)
                std = np.sqrt(self.reward_var / max(1, self.reward_count - 1)) + 1e-8
                normalized_reward = (magnitude - self.reward_mean) / std
                return self.intrinsic_reward_scale * normalized_reward
            else:
                return self.intrinsic_reward_scale * magnitude
    
    def train_cfn(self, state_batch, coin_flip_batch):
        """Train the coin flip prediction network."""
        predicted_flips = self.cfn_network(state_batch)
        coin_flip_tensor = torch.from_numpy(coin_flip_batch).float().to(self.device)
        
        # MSE loss between predicted and target coin flips
        loss = F.mse_loss(predicted_flips, coin_flip_tensor)
        
        self.cfn_optimizer.zero_grad()
        loss.backward()
        self.cfn_optimizer.step()
        
        return loss.item()
    
    def update_target(self):
        """Soft update of target networks for BOTH critics."""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_exploration_q_funcs.parameters(), 
                                               self.exploration_q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)
            
            for target_q_param, q_param in zip(self.target_task_q_funcs.parameters(), 
                                               self.task_q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)
    
    def train(self, src_replay_buffer, tar_replay_buffer, initial_state, batch_size=128, writer=None):
        self.total_it += 1
        if src_replay_buffer.size < batch_size or tar_replay_buffer.size < batch_size:
            return
        
        # --- 1. Update Domain Classifier ---
        if self.total_it > self.config.get('darc_warmup_steps', 10000):
            if self.total_it % self.config.get('classifier_update_freq', 10) == 0:
                self.update_classifier(src_replay_buffer, tar_replay_buffer, batch_size, writer)
        
        # --- 2. Exploration Policy Training (with CFN intrinsic rewards) ---
        tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(batch_size)
        
        # Generate coin flips for training CFN (store labels for supervised learning)
        coin_flips = []
        for i in range(batch_size):
            coin_flip = self.coin_flip_maker()
            coin_flips.append(coin_flip)
        
        coin_flips_batch = np.stack(coin_flips, axis=0)
        
        # Compute intrinsic rewards from prediction magnitude
        intrinsic_rewards = []
        for i in range(batch_size):
            intrinsic_reward = self.compute_intrinsic_reward(
                tar_state[i:i+1],
                update_stats=(i == 0)  # Only update stats once per batch
            )
            intrinsic_rewards.append(intrinsic_reward)
        
        intrinsic_rewards_tensor = torch.stack(intrinsic_rewards).to(self.device)
        
        with torch.no_grad():
            exploration_reward = tar_reward + intrinsic_rewards_tensor
            
            # Q-target for exploration policy
            next_action_exp, log_prob_exp, _ = self.exploration_policy(tar_next_state, get_logprob=True)
            q_t1, q_t2 = self.target_exploration_q_funcs(tar_next_state, next_action_exp)
            q_target = torch.min(q_t1, q_t2)
            value_target_exp = exploration_reward + tar_not_done * self.discount * \
                             (q_target - self.alpha.detach() * log_prob_exp)
        
        # Update exploration critic
        q1_exp, q2_exp = self.exploration_q_funcs(tar_state, tar_action)
        q_loss_exp = F.mse_loss(q1_exp, value_target_exp) + F.mse_loss(q2_exp, value_target_exp)
        
        self.exploration_q_optimizer.zero_grad()
        q_loss_exp.backward()
        self.exploration_q_optimizer.step()
        
        # Update exploration policy
        if self.total_it % self.config.get('policy_freq', 2) == 0:
            for p in self.exploration_q_funcs.parameters(): 
                p.requires_grad = False
            
            pi_action_exp, log_prob_pi_exp, _ = self.exploration_policy(tar_state, get_logprob=True)
            q1_pi_exp, q2_pi_exp = self.exploration_q_funcs(tar_state, pi_action_exp)
            policy_loss_exp = (self.alpha.detach() * log_prob_pi_exp - torch.min(q1_pi_exp, q2_pi_exp)).mean()
            
            self.exploration_policy_optimizer.zero_grad()
            policy_loss_exp.backward()
            self.exploration_policy_optimizer.step()
            
            for p in self.exploration_q_funcs.parameters(): 
                p.requires_grad = True
        
        # --- 3. Train CFN Network ---
        if self.total_it % self.config.get('cfn_update_freq', 1) == 0:
            cfn_loss = self.train_cfn(tar_state, coin_flips_batch)
            
            if writer is not None and self.total_it % 1000 == 0:
                writer.add_scalar('train/cfn_loss', cfn_loss, global_step=self.total_it)
                writer.add_scalar('train/intrinsic_reward_mean', intrinsic_rewards_tensor.mean().item(), 
                                global_step=self.total_it)
        
        # --- 4. Task Policy Training (with IS weighting) ---
        src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample(batch_size)
        tar_state_is, tar_action_is, tar_next_state_is, tar_reward_is, tar_not_done_is = \
            tar_replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            # Q-target for source data
            next_action_src, log_prob_src, _ = self.policy(src_next_state, get_logprob=True)
            q_t1_src, q_t2_src = self.target_task_q_funcs(src_next_state, next_action_src)
            value_target_src = src_reward + src_not_done * self.discount * \
                             (torch.min(q_t1_src, q_t2_src) - self.alpha * log_prob_src)
            
            # Q-target for target data
            next_action_tar, log_prob_tar, _ = self.policy(tar_next_state_is, get_logprob=True)
            q_t1_tar, q_t2_tar = self.target_task_q_funcs(tar_next_state_is, next_action_tar)
            value_target_tar = tar_reward_is + tar_not_done_is * self.discount * \
                             (torch.min(q_t1_tar, q_t2_tar) - self.alpha * log_prob_tar)
            
            # DARC importance weights for source data
            sas_probs_src, sa_probs_src = self.classifier(src_state, src_action, src_next_state, with_noise=False)
            darc_log_ratio_src = torch.log(sas_probs_src[:, 1:] + 1e-8) - torch.log(sas_probs_src[:, :1] + 1e-8) - \
                                (torch.log(sa_probs_src[:, 1:] + 1e-8) - torch.log(sa_probs_src[:, :1] + 1e-8))
            is_weights_src = torch.exp(darc_log_ratio_src.squeeze(-1)).detach()
            is_weights_src = torch.clamp(is_weights_src, 1e-4, 1.0)
            
            # Policy IS weights for target data
            log_prob_task_policy = self.policy.log_prob(tar_state_is, tar_action_is)
            log_prob_exp_policy = self.exploration_policy.log_prob(tar_state_is, tar_action_is)
            is_weights_tar = torch.exp((log_prob_task_policy - log_prob_exp_policy).squeeze(-1)).detach()
            is_weights_tar = torch.clamp(is_weights_tar, 1e-4, 1.0)
        
        # Update task critic
        q1_src, q2_src = self.task_q_funcs(src_state, src_action)
        q_loss_src = (is_weights_src * (F.mse_loss(q1_src, value_target_src, reduction='none') + 
                                        F.mse_loss(q2_src, value_target_src, reduction='none')).squeeze(-1)).mean()
        
        q1_tar, q2_tar = self.task_q_funcs(tar_state_is, tar_action_is)
        q_loss_tar = (is_weights_tar * (F.mse_loss(q1_tar, value_target_tar, reduction='none') + 
                                        F.mse_loss(q2_tar, value_target_tar, reduction='none')).squeeze(-1)).mean()
        
        total_q_loss_task = q_loss_src + q_loss_tar
        self.task_q_optimizer.zero_grad()
        total_q_loss_task.backward()
        self.task_q_optimizer.step()
        
        # Update task policy (uniform sampling - no IS weights)
        if self.total_it % self.config.get('policy_freq', 2) == 0:
            for p in self.task_q_funcs.parameters(): 
                p.requires_grad = False
            
            # Concatenate source and target states for policy update
            states_combined = torch.cat([src_state, tar_state_is], dim=0)
            
            # Policy gradient on combined batch (no importance weights)
            pi_action_combined, log_prob_pi_combined, _ = self.policy(states_combined, get_logprob=True)
            q1_pi_combined, q2_pi_combined = self.task_q_funcs(states_combined, pi_action_combined)
            min_q_pi_combined = torch.min(q1_pi_combined, q2_pi_combined)
            total_policy_loss = (self.alpha * log_prob_pi_combined - min_q_pi_combined).mean()
            
            # Temperature loss (on combined batch)
            temp_loss = -self.alpha * (log_prob_pi_combined.detach() + self.target_entropy).mean()
            
            self.policy_optimizer.zero_grad()
            total_policy_loss.backward()
            self.policy_optimizer.step()
            
            if self.config['temperature_opt']:
                self.temp_optimizer.zero_grad()
                temp_loss.backward()
                self.temp_optimizer.step()
            
            for p in self.task_q_funcs.parameters(): 
                p.requires_grad = True
            
            self.update_target()
            
            if writer is not None and self.total_it % 1000 == 0:
                writer.add_scalar('train/task_policy_loss', total_policy_loss.item(), global_step=self.total_it)
                writer.add_scalar('train/task_q_loss', total_q_loss_task.item(), global_step=self.total_it)
                writer.add_scalar('train/exploration_policy_loss', policy_loss_exp.item(), global_step=self.total_it)
                writer.add_scalar('train/exploration_q_loss', q_loss_exp.item(), global_step=self.total_it)
                writer.add_scalar('train/alpha', self.alpha.item(), global_step=self.total_it)
    
    def save(self, filename):
        torch.save({
            'policy': self.policy.state_dict(),
            'exploration_policy': self.exploration_policy.state_dict(),
            'task_q_funcs': self.task_q_funcs.state_dict(),
            'exploration_q_funcs': self.exploration_q_funcs.state_dict(),
            'classifier': self.classifier.state_dict(),
            'cfn_network': self.cfn_network.state_dict(),
            'log_alpha': self.log_alpha,
            'total_it': self.total_it,
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.exploration_policy.load_state_dict(checkpoint['exploration_policy'])
        self.task_q_funcs.load_state_dict(checkpoint['task_q_funcs'])
        self.exploration_q_funcs.load_state_dict(checkpoint['exploration_q_funcs'])
        self.classifier.load_state_dict(checkpoint['classifier'])
        self.cfn_network.load_state_dict(checkpoint['cfn_network'])
        self.log_alpha = checkpoint['log_alpha']
        self.total_it = checkpoint['total_it']
        
        self.target_task_q_funcs = copy.deepcopy(self.task_q_funcs)
        self.target_exploration_q_funcs = copy.deepcopy(self.exploration_q_funcs)
    
    @property
    def actor(self):
        return self.policy
