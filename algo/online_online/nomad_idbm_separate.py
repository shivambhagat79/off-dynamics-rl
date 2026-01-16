import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, constraints
from torch.distributions.transforms import Transform

# Import the new models for LIBERTY exploration
from liberty_models import MetricModel, InverseDynamicsModel, ForwardDynamicsModel


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


class NOMAD_IDBM_SEPARATE(object):
    """
    NOMAD-IDBM with SEPARATE critics for exploration and task policies.
    
    Key differences from original:
    - exploration_q_funcs: Dedicated critic for exploration policy (with intrinsic rewards)
    - task_q_funcs: Dedicated critic for task policy (with IS-weighted extrinsic rewards)
    - No gradient interference between exploration and task objectives
    - Cleaner separation of concerns at the cost of 2x critic parameters
    """

    def __init__(self,
                 config,
                 device,
                 target_entropy=None,
                 ):
        self.config = config
        self.device = device
        self.discount = config['gamma']
        self.tau = config['tau']
        self.target_entropy = target_entropy if target_entropy else -config['action_dim']

        self.total_it = 0

        # Hyperparameters for new features
        self.use_liberty_reward = config.get('use_liberty_reward', False)
        self.use_only_policy_is = config.get('use_only_policy_is', False)
        self.temp = config.get('temp', 0.2)
        self.temp_lr = config.get('temp_lr', 1e-5)
        self.policy_noise_clip = config.get('policy_noise_clip', 0.5)

        # === SEPARATE CRITICS ===
        # Exploration Q-functions (for target_policy with intrinsic rewards)
        self.exploration_q_funcs = DoubleQFunc(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.target_exploration_q_funcs = copy.deepcopy(self.exploration_q_funcs)
        self.target_exploration_q_funcs.eval()
        for p in self.target_exploration_q_funcs.parameters():
            p.requires_grad = False

        # Task Q-functions (for source policy with extrinsic rewards)
        self.task_q_funcs = DoubleQFunc(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.target_task_q_funcs = copy.deepcopy(self.task_q_funcs)
        self.target_task_q_funcs.eval()
        for p in self.target_task_q_funcs.parameters():
            p.requires_grad = False

        # Source policy (actor) - for training and testing
        self.policy = Policy(config['state_dim'], config['action_dim'], config['max_action'], hidden_size=config['hidden_sizes']).to(self.device)

        # Target policy (exploration policy) - for exploration in target env
        self.target_policy = Policy(config['state_dim'], config['action_dim'], config['max_action'], hidden_size=config['hidden_sizes']).to(self.device)
        self.exploration_mode = False  # Flag to switch to target_policy for action selection

        # Temperature (alpha) for SAC
        if config['temperature_opt']:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        else:
            self.log_alpha = torch.log(torch.FloatTensor([0.2])).to(self.device)

        # DARC domain classifier
        self.classifier = Classifier(config['state_dim'], config['action_dim'], config['hidden_sizes'], config['gaussian_noise_std']).to(self.device)

        # LIBERTY Exploration Components
        self.metric_model = MetricModel(config['state_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.inverse_model = InverseDynamicsModel(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.forward_model = ForwardDynamicsModel(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.intrinsic_reward_coef = config.get('intrinsic_reward_coef', 0.003)

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

    def update_classifier(self, src_replay_buffer, tar_replay_buffer, batch_size, writer=None):
        src_state, src_action, src_next_state, _, _ = src_replay_buffer.sample(batch_size)
        tar_state, tar_action, tar_next_state, _, _ = tar_replay_buffer.sample(batch_size)
        state = torch.cat([src_state, tar_state], 0)
        action = torch.cat([src_action, tar_action], 0)
        next_state = torch.cat([src_next_state, tar_next_state], 0)
        label = torch.cat([torch.zeros(size=(batch_size,)), torch.ones(size=(batch_size,))], dim=0).long().to(self.device)

        # Shuffle batch
        indices = torch.randperm(label.shape[0])
        state_batch, action_batch, nextstate_batch, label_batch = state[indices], action[indices], next_state[indices], label[indices]

        sas_logits, sa_logits = self.classifier(state_batch, action_batch, nextstate_batch, with_noise=True)
        loss = F.cross_entropy(sas_logits, label_batch) + F.cross_entropy(sa_logits, label_batch)

        self.classifier_optimizer.zero_grad()
        loss.backward()
        self.classifier_optimizer.step()

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/classifier_loss', loss.item(), global_step=self.total_it)

    def _update_liberty_models(self, src_data, tar_data, writer):
        # Unpack both Source and Target data
        src_state, src_action, src_reward, src_next_state = src_data
        tar_state, tar_action, tar_reward, tar_next_state = tar_data

        # Update inverse and forward dynamics models on Target data
        pred_action = self.inverse_model(tar_state, tar_next_state)
        inverse_loss = F.mse_loss(pred_action, tar_action)

        pred_next_state_mu, pred_next_state_std = self.forward_model(tar_state, tar_action)
        forward_dist = Normal(pred_next_state_mu, pred_next_state_std)
        forward_loss = -forward_dist.log_prob(tar_next_state).mean()

        dynamics_loss = inverse_loss + forward_loss
        self.dynamics_optimizer.zero_grad()
        dynamics_loss.backward()
        self.dynamics_optimizer.step()

        # Update metric model - compare Source states to Target states for Bisimulation
        state, action, reward, next_state = src_state, src_action, src_reward, src_next_state
        state_p, action_p, reward_p, next_state_p = tar_state, tar_action, tar_reward, tar_next_state

        with torch.no_grad():
            # Reward difference across domains
            r_dist = torch.abs(reward - reward_p)

            # Forward dynamics (Wasserstein) difference
            mu, std = self.forward_model(state, action)
            mu_p, std_p = self.forward_model(state_p, action_p)

            w2_dist_sq = torch.sum((mu - mu_p)**2, dim=-1, keepdim=True) + torch.sum((std - std_p)**2, dim=-1, keepdim=True)
            w2_dist = torch.sqrt(w2_dist_sq + 1e-6)

            # Inverse dynamics difference
            pred_a = self.inverse_model(state, next_state)
            pred_a_p = self.inverse_model(state_p, next_state_p)
            inv_dist = torch.sum(torch.abs(pred_a - pred_a_p), dim=-1, keepdim=True)

            target_metric = r_dist + self.discount * (w2_dist + inv_dist)

        predicted_metric = self.metric_model(state, state_p)
        metric_loss = F.mse_loss(predicted_metric, target_metric)
        self.metric_optimizer.zero_grad()
        metric_loss.backward()
        self.metric_optimizer.step()

        if writer is not None and self.total_it % 1000 == 0:
            writer.add_scalar('liberty/dynamics_loss', dynamics_loss.item(), self.total_it)
            writer.add_scalar('liberty/metric_loss', metric_loss.item(), self.total_it)

    def update_target(self):
        """Soft update of target networks for BOTH critics."""
        with torch.no_grad():
            # Update exploration critic target
            for target_q_param, q_param in zip(self.target_exploration_q_funcs.parameters(), self.exploration_q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)
            
            # Update task critic target
            for target_q_param, q_param in zip(self.target_task_q_funcs.parameters(), self.task_q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def train(self, src_replay_buffer, tar_replay_buffer, initial_state, batch_size=128, writer=None):
        self.total_it += 1
        if src_replay_buffer.size < batch_size or tar_replay_buffer.size < batch_size:
            return

        # --- 1. Auxiliary Model Updates ---
        src_state_lib, src_action_lib, src_next_state_lib, src_reward_lib, _ = src_replay_buffer.sample(batch_size)
        tar_state_lib, tar_action_lib, tar_next_state_lib, tar_reward_lib, _ = tar_replay_buffer.sample(batch_size)

        src_data = (src_state_lib, src_action_lib, src_reward_lib, src_next_state_lib)
        tar_data = (tar_state_lib, tar_action_lib, tar_reward_lib, tar_next_state_lib)

        self._update_liberty_models(src_data, tar_data, writer)

        if self.total_it > self.config.get('darc_warmup_steps', 10000):
            if self.total_it % self.config.get('classifier_update_freq', 10) == 0:
                self.update_classifier(src_replay_buffer, tar_replay_buffer, batch_size, writer)

        # --- 2. Exploration Policy Training Step ---
        # Uses EXPLORATION CRITIC with intrinsic rewards
        tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Add intrinsic reward for the exploratory policy
            if initial_state is not None:
                initial_state_batch = torch.Tensor(initial_state).expand_as(tar_state).to(self.device)
                potential_s = self.metric_model(tar_state, initial_state_batch)
                potential_s_next = self.metric_model(tar_next_state, initial_state_batch)
                intrinsic_reward = self.discount * potential_s_next - potential_s
                tar_reward += self.intrinsic_reward_coef * intrinsic_reward

            # Compute target Q value using EXPLORATION critic
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

        # Update exploratory policy using EXPLORATION critic
        if self.total_it % self.config.get('policy_freq', 2) == 0:
            for p in self.exploration_q_funcs.parameters(): p.requires_grad = False
            pi_action_tar, log_prob_pi_tar, _ = self.target_policy(tar_state, get_logprob=True)
            q1_pi, q2_pi = self.exploration_q_funcs(tar_state, pi_action_tar)
            policy_loss_exploration = (self.alpha.detach() * log_prob_pi_tar - torch.min(q1_pi, q2_pi)).mean()
            self.target_policy_optimizer.zero_grad()
            policy_loss_exploration.backward()
            self.target_policy_optimizer.step()
            for p in self.exploration_q_funcs.parameters(): p.requires_grad = True

        # --- 3. Task Policy Training Step ---
        # Uses TASK CRITIC with extrinsic rewards and importance sampling
        src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample(batch_size)
        tar_state_is, tar_action_is, tar_next_state_is, tar_reward_is, tar_not_done_is = tar_replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Q-target for source data using TASK critic
            next_action_src, log_prob_src, _ = self.policy(src_next_state, get_logprob=True)
            noise = (torch.randn_like(next_action_src) * self.temp).clamp(-self.policy_noise_clip, self.policy_noise_clip)
            noisy_next_action_src = (next_action_src + noise).clamp(-self.config['max_action'], self.config['max_action'])
            q_t1_src, q_t2_src = self.target_task_q_funcs(src_next_state, noisy_next_action_src)
            value_target_src = src_reward + src_not_done * self.discount * (torch.min(q_t1_src, q_t2_src) - self.alpha * log_prob_src)

            # Q-target for target data using TASK critic
            next_action_tar_is, log_prob_tar_is, _ = self.policy(tar_next_state_is, get_logprob=True)
            noise_is = (torch.randn_like(next_action_tar_is) * self.temp).clamp(-self.policy_noise_clip, self.policy_noise_clip)
            noisy_next_action_tar_is = (next_action_tar_is + noise_is).clamp(-self.config['max_action'], self.config['max_action'])
            q_t1_tar_is, q_t2_tar_is = self.target_task_q_funcs(tar_next_state_is, noisy_next_action_tar_is)
            value_target_tar_is = tar_reward_is + tar_not_done_is * self.discount * (torch.min(q_t1_tar_is, q_t2_tar_is) - self.alpha * log_prob_tar_is)

            # Importance Sampling Weights for Source Data (DARC weights)
            sas_probs_src, sa_probs_src = self.classifier(src_state, src_action, src_next_state, with_noise=False)
            darc_log_ratio_src = torch.log(sas_probs_src[:, 1:] + 1e-8) - torch.log(sas_probs_src[:, :1] + 1e-8) - \
                                 (torch.log(sa_probs_src[:, 1:] + 1e-8) - torch.log(sa_probs_src[:, :1] + 1e-8))
            is_weights_src = torch.exp(darc_log_ratio_src.squeeze(-1)).detach()
            is_weights_src = torch.clamp(is_weights_src, 1e-4, 1.0)

            # Importance Sampling Weights for Target Data (Policy weights)
            log_prob_src_policy = self.policy.log_prob(tar_state_is, tar_action_is)
            log_prob_tar_policy = self.target_policy.log_prob(tar_state_is, tar_action_is)
            log_w_policy = (log_prob_src_policy - log_prob_tar_policy).squeeze(-1)
            is_weights_tar = torch.exp(log_w_policy).detach()
            is_weights_tar = torch.clamp(is_weights_tar, 1e-4, 1.0)

            if writer is not None and self.total_it % 1000 == 0:
                writer.add_scalar('nomad_separate/is_weights_src_mean', is_weights_src.mean().item(), self.total_it)
                writer.add_scalar('nomad_separate/is_weights_tar_mean', is_weights_tar.mean().item(), self.total_it)
                writer.add_scalar('nomad_separate/action_noise_mean', noise.abs().mean().item(), self.total_it)
                writer.add_scalar('nomad_separate/temperature', self.temp, self.total_it)
                writer.add_scalar('nomad_separate/exploration_q_loss', q_loss_exploration.item(), self.total_it)

        # Update TASK critic
        q1_src, q2_src = self.task_q_funcs(src_state, src_action)
        q_loss_src_unweighted = F.mse_loss(q1_src, value_target_src, reduction='none') + F.mse_loss(q2_src, value_target_src, reduction='none')
        q_loss_src = (q_loss_src_unweighted.squeeze(-1) * is_weights_src).mean()

        q1_tar_is, q2_tar_is = self.task_q_funcs(tar_state_is, tar_action_is)
        q_loss_tar_is_unweighted = F.mse_loss(q1_tar_is, value_target_tar_is, reduction='none') + F.mse_loss(q2_tar_is, value_target_tar_is, reduction='none')
        q_loss_tar_is = (q_loss_tar_is_unweighted.squeeze(-1) * is_weights_tar).mean()

        total_q_loss_task = q_loss_src + q_loss_tar_is
        self.task_q_optimizer.zero_grad()
        total_q_loss_task.backward()
        self.task_q_optimizer.step()

        # Update task policy using TASK critic
        if self.total_it % self.config.get('policy_freq', 2) == 0:
            self.temp *= (1 - self.temp_lr)  # Decay temperature
            for p in self.task_q_funcs.parameters(): p.requires_grad = False

            # Policy loss from source batch (with DARC IS)
            pi_action_src, log_prob_pi_src, _ = self.policy(src_state, get_logprob=True)
            q1_pi_src, q2_pi_src = self.task_q_funcs(src_state, pi_action_src)
            policy_loss_src_unweighted = (self.alpha * log_prob_pi_src - torch.min(q1_pi_src, q2_pi_src))
            policy_loss_src = (policy_loss_src_unweighted.squeeze(-1) * is_weights_src).mean()

            # Policy loss from target batch (with Policy IS)
            pi_action_tar_is, log_prob_pi_tar_is, _ = self.policy(tar_state_is, get_logprob=True)
            q1_pi_tar_is, q2_pi_tar_is = self.task_q_funcs(tar_state_is, pi_action_tar_is)
            policy_loss_tar_unweighted = self.alpha * log_prob_pi_tar_is - torch.min(q1_pi_tar_is, q2_pi_tar_is)
            policy_loss_tar_is = (policy_loss_tar_unweighted.squeeze(-1) * is_weights_tar).mean()

            total_policy_loss = policy_loss_src + policy_loss_tar_is

            # Temperature loss
            temp_loss = -self.alpha * (log_prob_pi_src.detach() + self.target_entropy).mean()

            self.policy_optimizer.zero_grad()
            total_policy_loss.backward()
            self.policy_optimizer.step()

            if self.config['temperature_opt']:
                self.temp_optimizer.zero_grad()
                temp_loss.backward()
                self.temp_optimizer.step()

            for p in self.task_q_funcs.parameters(): p.requires_grad = True

            # Update both target networks
            self.update_target()

            if writer is not None:
                writer.add_scalar('nomad_separate/task_q_loss', total_q_loss_task.item(), self.total_it)
                writer.add_scalar('nomad_separate/task_policy_loss', total_policy_loss.item(), self.total_it)
                writer.add_scalar('nomad_separate/exploration_policy_loss', policy_loss_exploration.item(), self.total_it)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def save(self, filename):
        torch.save(self.exploration_q_funcs.state_dict(), filename + "_exploration_critic")
        torch.save(self.exploration_q_optimizer.state_dict(), filename + "_exploration_critic_optimizer")
        torch.save(self.task_q_funcs.state_dict(), filename + "_task_critic")
        torch.save(self.task_q_optimizer.state_dict(), filename + "_task_critic_optimizer")
        torch.save(self.policy.state_dict(), filename + "_actor")
        torch.save(self.policy_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.target_policy.state_dict(), filename + "_exploration_actor")
        torch.save(self.target_policy_optimizer.state_dict(), filename + "_exploration_actor_optimizer")
        torch.save(self.classifier.state_dict(), filename + "_classifier")
        torch.save(self.classifier_optimizer.state_dict(), filename + "_classifier_optimizer")
        torch.save(self.metric_model.state_dict(), filename + "_metric_model")
        torch.save(self.dynamics_optimizer.state_dict(), filename + "_dynamics_optimizer")

    def load(self, filename):
        self.exploration_q_funcs.load_state_dict(torch.load(filename + "_exploration_critic"))
        self.exploration_q_optimizer.load_state_dict(torch.load(filename + "_exploration_critic_optimizer"))
        self.task_q_funcs.load_state_dict(torch.load(filename + "_task_critic"))
        self.task_q_optimizer.load_state_dict(torch.load(filename + "_task_critic_optimizer"))
        self.policy.load_state_dict(torch.load(filename + "_actor"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.target_policy.load_state_dict(torch.load(filename + "_exploration_actor"))
        self.target_policy_optimizer.load_state_dict(torch.load(filename + "_exploration_actor_optimizer"))
        self.classifier.load_state_dict(torch.load(filename + "_classifier"))
        self.classifier_optimizer.load_state_dict(torch.load(filename + "_classifier_optimizer"))
        self.metric_model.load_state_dict(torch.load(filename + "_metric_model"))
        self.dynamics_optimizer.load_state_dict(torch.load(filename + "_dynamics_optimizer"))
