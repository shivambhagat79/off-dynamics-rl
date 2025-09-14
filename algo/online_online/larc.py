import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, constraints
from torch.distributions.transforms import Transform

from liberty_models import MetricModel, InverseDynamicsModel, ForwardDynamicsModel


class TanhTransform(Transform):
    """
    Transform via the mapping :math:`y = \\tanh(x)`.
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


class DoubleQFunc(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DoubleQFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)


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


class LARC(object):
    def __init__(self, config, device, target_entropy=None):
        self.config = config
        self.device = device
        self.discount = config['gamma']
        self.tau = config['tau']
        self.target_entropy = target_entropy if target_entropy else -config['action_dim']
        self.total_it = 0

        # Main policy and Q-functions (for DARC on source env)
        self.policy = Policy(config['state_dim'], config['action_dim'], config['max_action'], hidden_size=config['hidden_sizes']).to(self.device)
        self.q_funcs = DoubleQFunc(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()

        # *** IMPROVEMENT 1: Decoupled Q-functions for exploratory policy ***
        # Exploratory policy and its own Q-functions (for LIBERTY on target env)
        self.explore_policy = Policy(config['state_dim'], config['action_dim'], config['max_action'], hidden_size=config['hidden_sizes']).to(self.device)
        self.explore_q_funcs = DoubleQFunc(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.explore_target_q_funcs = copy.deepcopy(self.explore_q_funcs)
        self.explore_target_q_funcs.eval()

        if config['temperature_opt']:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        else:
            self.log_alpha = torch.log(torch.FloatTensor([0.2])).to(self.device)

        # DARC components
        self.classifier = Classifier(config['state_dim'], config['action_dim'], config['hidden_sizes'], config['gaussian_noise_std']).to(self.device)

        # LIBERTY components
        self.metric_model = MetricModel(config['state_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.inverse_model = InverseDynamicsModel(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.forward_model = ForwardDynamicsModel(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['actor_lr'])
        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=config['critic_lr'])

        self.explore_policy_optimizer = torch.optim.Adam(self.explore_policy.parameters(), lr=config['actor_lr'])
        self.explore_q_optimizer = torch.optim.Adam(self.explore_q_funcs.parameters(), lr=config['critic_lr'])

        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=config['actor_lr'])
        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=config['actor_lr'])
        self.metric_optimizer = torch.optim.Adam(self.metric_model.parameters(), lr=config['liberty_lr'])
        self.dynamics_optimizer = torch.optim.Adam(list(self.inverse_model.parameters()) + list(self.forward_model.parameters()), lr=config['liberty_lr'])

        # Hyperparameters for improvements
        self.intrinsic_reward_coef = config.get('intrinsic_reward_coef', 0.01)
        self.target_policy_train_ratio = config.get('target_policy_train_ratio', 10)
        # *** IMPROVEMENT 2: Frequency for updating auxiliary models ***
        self.aux_update_freq = config.get('aux_update_freq', 5)


    def select_action(self, state, test=True, explore=False):
        policy_to_use = self.explore_policy if explore and not test else self.policy
        with torch.no_grad():
            action, _, mean = policy_to_use(torch.Tensor(state).view(1, -1).to(self.device))

        if test:
            return mean.squeeze().cpu().numpy()
        else:
            return action.squeeze().cpu().numpy()

    def update_classifier(self, src_replay_buffer, tar_replay_buffer, batch_size, writer=None):
        # This function remains largely the same
        src_state, src_action, src_next_state, _, _ = src_replay_buffer.sample(batch_size)
        tar_state, tar_action, tar_next_state, _, _ = tar_replay_buffer.sample(batch_size)

        state = torch.cat([src_state, tar_state], 0)
        action = torch.cat([src_action, tar_action], 0)
        next_state = torch.cat([src_next_state, tar_next_state], 0)
        label = torch.cat([torch.zeros(size=(batch_size,)), torch.ones(size=(batch_size,))], dim=0).long().to(self.device)

        indices = torch.randperm(label.shape[0])
        state_batch, action_batch, nextstate_batch, label = state[indices], action[indices], next_state[indices], label[indices]

        sas_logits, sa_logits = self.classifier(state_batch, action_batch, nextstate_batch, with_noise=True)
        loss = F.cross_entropy(sas_logits, label) + F.cross_entropy(sa_logits, label)

        self.classifier_optimizer.zero_grad()
        loss.backward()
        self.classifier_optimizer.step()

        if writer is not None and self.total_it % 1000 == 0:
            writer.add_scalar('train/classifier_loss', loss.item(), self.total_it)

    def update_liberty_dynamics(self, tar_replay_buffer, batch_size, writer=None):
        # Update dynamics models (Forward and Inverse)
        state, action, next_state, _, _ = tar_replay_buffer.sample(batch_size)

        pred_action = self.inverse_model(state, next_state)
        inverse_loss = F.mse_loss(pred_action, action)
        pred_next_state_mu, pred_next_state_std = self.forward_model(state, action)
        forward_loss = -Normal(pred_next_state_mu, pred_next_state_std).log_prob(next_state).mean()
        dynamics_loss = inverse_loss + forward_loss

        self.dynamics_optimizer.zero_grad()
        dynamics_loss.backward()
        self.dynamics_optimizer.step()

        if writer is not None and self.total_it % 1000 == 0:
            writer.add_scalar('liberty/dynamics_loss', dynamics_loss.item(), self.total_it)

    def update_target_networks(self, main_net=True, explore_net=True):
        if main_net:
            with torch.no_grad():
                for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                    target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)
        if explore_net:
            with torch.no_grad():
                for target_q_param, q_param in zip(self.explore_target_q_funcs.parameters(), self.explore_q_funcs.parameters()):
                    target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def train(self, src_replay_buffer, tar_replay_buffer, initial_state, batch_size=256, writer=None):
        self.total_it += 1

        if src_replay_buffer.size < batch_size or tar_replay_buffer.size < batch_size:
            return

        # --- Train Source (Main) Policy with DARC ---
        src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample(batch_size)

        if self.total_it > self.config.get('darc_warmup_steps', 10000):
            if self.total_it % self.aux_update_freq == 0:
                self.update_classifier(src_replay_buffer, tar_replay_buffer, batch_size, writer)

            with torch.no_grad():
                sas_probs, sa_probs = self.classifier(src_state, src_action, src_next_state, with_noise=False)
                reward_penalty = (torch.log(sas_probs[:, 1:] + 1e-8) - torch.log(sa_probs[:, 1:] + 1e-8)) - \
                                 (torch.log(sas_probs[:, :1] + 1e-8) - torch.log(sa_probs[:, :1] + 1e-8))
                src_reward += self.config['penalty_coefficient'] * reward_penalty

        # Q-function update for main policy
        with torch.no_grad():
            next_action, log_prob, _ = self.policy(src_next_state, get_logprob=True)
            target_q1, target_q2 = self.target_q_funcs(src_next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * log_prob
            target_q = src_reward + src_not_done * self.discount * target_q

        current_q1, current_q2 = self.q_funcs(src_state, src_action)
        q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Main policy and temperature update
        for p in self.q_funcs.parameters(): p.requires_grad = False
        action, log_prob, _ = self.policy(src_state, get_logprob=True)
        q1_pi, q2_pi = self.q_funcs(src_state, action)
        q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha * log_prob - q_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        for p in self.q_funcs.parameters(): p.requires_grad = True

        if self.config['temperature_opt']:
            # Temperature is shared, updated based on main policy's entropy
            temp_loss = -self.alpha * (log_prob.detach() + self.target_entropy).mean()
            self.temp_optimizer.zero_grad()
            temp_loss.backward()
            self.temp_optimizer.step()

        self.update_target_networks(main_net=True, explore_net=False)

        # --- Train Target (Exploratory) Policy with LIBERTY ---
        # Delayed update for liberty dynamics models
        if self.total_it % self.aux_update_freq == 0:
            self.update_liberty_dynamics(tar_replay_buffer, batch_size, writer)

        for _ in range(self.target_policy_train_ratio):
            tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(batch_size)

            # Update metric model (needs to be fast to guide policy)
            perm = torch.randperm(tar_state.size(0))
            state_p, reward_p = tar_state[perm], tar_reward[perm]
            with torch.no_grad():
                r_dist = F.mse_loss(tar_reward, reward_p, reduction='none')
            predicted_metric = self.metric_model(tar_state, state_p) # Simplified for speed
            metric_loss = F.mse_loss(predicted_metric, r_dist)
            self.metric_optimizer.zero_grad(); metric_loss.backward(); self.metric_optimizer.step()

            # Q-function update for exploratory policy
            with torch.no_grad():
                initial_state_batch = torch.Tensor(initial_state).expand_as(tar_state).to(self.device)
                potential_s_next = self.metric_model(tar_next_state, initial_state_batch)
                potential_s = self.metric_model(tar_state, initial_state_batch)
                intrinsic_reward = self.discount * potential_s_next - potential_s

                next_explore_action, next_log_prob, _ = self.explore_policy(tar_next_state, get_logprob=True)
                exp_target_q1, exp_target_q2 = self.explore_target_q_funcs(tar_next_state, next_explore_action)
                exp_target_q = torch.min(exp_target_q1, exp_target_q2) - self.alpha.detach() * next_log_prob
                total_reward = tar_reward + self.intrinsic_reward_coef * intrinsic_reward
                exp_target_q = total_reward + tar_not_done * self.discount * exp_target_q

            exp_current_q1, exp_current_q2 = self.explore_q_funcs(tar_state, tar_action)
            exp_q_loss = F.mse_loss(exp_current_q1, exp_target_q) + F.mse_loss(exp_current_q2, exp_target_q)

            self.explore_q_optimizer.zero_grad(); exp_q_loss.backward(); self.explore_q_optimizer.step()

            # Exploratory policy update
            for p in self.explore_q_funcs.parameters(): p.requires_grad = False
            explore_action, log_prob, _ = self.explore_policy(tar_state, get_logprob=True)
            exp_q1, exp_q2 = self.explore_q_funcs(tar_state, explore_action)
            exp_q_pi = torch.min(exp_q1, exp_q2)
            explore_policy_loss = (self.alpha.detach() * log_prob - exp_q_pi).mean()

            self.explore_policy_optimizer.zero_grad(); explore_policy_loss.backward(); self.explore_policy_optimizer.step()
            for p in self.explore_q_funcs.parameters(): p.requires_grad = True

        self.update_target_networks(main_net=False, explore_net=True)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def save(self, filename):
        torch.save(self.policy.state_dict(), filename + "_main_policy")
        torch.save(self.q_funcs.state_dict(), filename + "_main_critic")
        torch.save(self.explore_policy.state_dict(), filename + "_explore_policy")
        torch.save(self.explore_q_funcs.state_dict(), filename + "_explore_critic")
        # ... save other components if needed

    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename + "_main_policy"))
        self.q_funcs.load_state_dict(torch.load(filename + "_main_critic"))
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.explore_policy.load_state_dict(torch.load(filename + "_explore_policy"))
        self.explore_q_funcs.load_state_dict(torch.load(filename + "_explore_critic"))
        self.explore_target_q_funcs = copy.deepcopy(self.explore_q_funcs)
        # ... load other components

