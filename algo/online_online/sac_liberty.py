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
    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
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
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
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


class SAC_LIBERTY(object):

    def __init__(self,
                 config,
                 device,
                 target_entropy=None,
                 ):
        self.config=  config
        self.device = device
        self.discount = config['gamma']
        self.tau = config['tau']
        self.target_entropy = target_entropy if target_entropy else -config['action_dim']
        self.update_interval = config['update_interval']

        self.total_it = 0

        # aka critic
        self.q_funcs = DoubleQFunc(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # aka actor
        self.policy = Policy(config['state_dim'], config['action_dim'], config['max_action'], hidden_size=config['hidden_sizes']).to(self.device)

        # aka temperature
        if config['temperature_opt']:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        else:
            self.log_alpha = torch.log(torch.FloatTensor([0.2])).to(self.device)

        # --- LIBERTY Exploration Components ---

        self.metric_model = MetricModel(config['state_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.inverse_model = InverseDynamicsModel(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.forward_model = ForwardDynamicsModel(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)

        self.metric_optimizer = torch.optim.Adam(self.metric_model.parameters(), lr=config['liberty_lr'])
        # Both dynamics models can be trained with a single optimizer
        self.dynamics_optimizer = torch.optim.Adam(
            list(self.inverse_model.parameters()) + list(self.forward_model.parameters()),
            lr=config['liberty_lr']
        )
        self.intrinsic_reward_coef = config['intrinsic_reward_coef']
        # --- End LIBERTY Components ---

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=config['critic_lr'])
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['actor_lr'])
        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=config['actor_lr'])

    def select_action(self, state, test=True):
        with torch.no_grad():
            action, _, mean = self.policy(torch.Tensor(state).view(1,-1).to(self.device))
        if test:
            return mean.squeeze().cpu().numpy()
        else:
            return action.squeeze().cpu().numpy()

    def update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def update_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, not_done_batch, writer=None):
        with torch.no_grad():
            nextaction_batch, logprobs_batch, _ = self.policy(nextstate_batch, get_logprob=True)
            q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
            # take min to mitigate positive bias in q-function training
            q_target = torch.min(q_t1, q_t2)
            value_target = reward_batch + not_done_batch * self.discount * (q_target - self.alpha * logprobs_batch)
        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/q1', q_1.mean(), self.total_it)
            writer.add_scalar('train/logprob', logprobs_batch.mean(), self.total_it)
        loss = F.mse_loss(q_1, value_target) + F.mse_loss(q_2, value_target)
        return loss

    def update_policy_and_temp(self, state_batch):
        action_batch, logprobs_batch, _ = self.policy(state_batch, get_logprob=True)
        q_b1, q_b2 = self.q_funcs(state_batch, action_batch)
        qval_batch = torch.min(q_b1, q_b2)
        policy_loss = (self.alpha * logprobs_batch - qval_batch).mean()
        temp_loss = -self.alpha * (logprobs_batch.detach() + self.target_entropy).mean()
        return policy_loss, temp_loss

    def _update_liberty_models(self, state, action, reward, next_state, writer):
        # Inverse dynamics loss
        pred_action = self.inverse_model(state, next_state)
        inverse_loss = F.mse_loss(pred_action, action)

        # Forward dynamics loss
        pred_next_state_mu, pred_next_state_std = self.forward_model(state, action)
        # Negative log-likelihood for Gaussian
        forward_loss = -Normal(pred_next_state_mu, pred_next_state_std).log_prob(next_state).mean()

        # Update both dynamics models
        self.dynamics_optimizer.zero_grad()
        (inverse_loss + forward_loss).backward()
        self.dynamics_optimizer.step()

        # Metric model loss (contrastive, as per paper)
        # Create a permuted batch for contrastive loss
        perm = torch.randperm(state.size(0))
        state_p, action_p, reward_p, next_state_p = state[perm], action[perm], reward[perm], next_state[perm]

        with torch.no_grad():
            r_dist = F.mse_loss(reward, reward_p, reduction='none')

            # Wasserstein distance between predicted next state distributions
            pred_mu, pred_std = self.forward_model(state, action)
            pred_mu_p, pred_std_p = self.forward_model(state_p, action_p)
            w2_dist_sq = F.mse_loss(pred_mu, pred_mu_p, reduction='none').sum(dim=-1, keepdim=True) + \
                         F.mse_loss(pred_std, pred_std_p, reduction='none').sum(dim=-1, keepdim=True)
            w2_dist = torch.sqrt(w2_dist_sq + 1e-6)

            # L1 distance between predicted actions from inverse model
            pred_a = self.inverse_model(state, next_state)
            pred_a_p = self.inverse_model(state_p, next_state_p)
            inv_dist = F.l1_loss(pred_a, pred_a_p, reduction='none').sum(dim=-1, keepdim=True)

            target_metric = r_dist + self.discount * w2_dist + self.discount * inv_dist

        predicted_metric = self.metric_model(state, state_p)
        metric_loss = F.mse_loss(predicted_metric, target_metric)

        self.metric_optimizer.zero_grad()
        metric_loss.backward()
        self.metric_optimizer.step()

        if writer is not None and self.total_it % 1000 == 0:
            writer.add_scalar('liberty/inverse_loss', inverse_loss.item(), self.total_it)
            writer.add_scalar('liberty/forward_loss', forward_loss.item(), self.total_it)
            writer.add_scalar('liberty/metric_loss', metric_loss.item(), self.total_it)

    def train(self, src_replay_buffer, tar_replay_buffer, initial_state, batch_size=128, writer=None):
        self.total_it += 1

        # Only train on target buffer if it has enough samples
        if tar_replay_buffer.size >= batch_size:
            tar_state, tar_action, tar_next_state, tar_reward, _ = tar_replay_buffer.sample(batch_size)
            self._update_liberty_models(tar_state, tar_action, tar_reward, tar_next_state, writer)

        # Standard SAC training
        if src_replay_buffer.size < batch_size or tar_replay_buffer.size < batch_size:
            return

        src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample(batch_size)
        tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(batch_size)

        # --- Add LIBERTY intrinsic reward to target transitions ONLY ---

        with torch.no_grad():
            # Potential Phi(s) is d(s, s_0)
            # We need a batch of initial states matching the tar_state batch size
            initial_state_batch = torch.Tensor(initial_state).expand_as(tar_state).to(self.device)

            potential_s = self.metric_model(tar_state, initial_state_batch)
            potential_s_next = self.metric_model(tar_next_state, initial_state_batch)

            # Shaping reward F = gamma * Phi(s') - Phi(s)
            intrinsic_reward = self.discount * potential_s_next - potential_s

            if writer is not None and self.total_it % 1000 == 0:
                writer.add_scalar('liberty/intrinsic_reward_mean', intrinsic_reward.mean().item(), self.total_it)

            tar_reward += self.intrinsic_reward_coef * intrinsic_reward
        # --- End intrinsic reward calculation ---

        state = torch.cat([src_state, tar_state], 0)
        action = torch.cat([src_action, tar_action], 0)
        next_state = torch.cat([src_next_state, tar_next_state], 0)
        reward = torch.cat([src_reward, tar_reward], 0)
        not_done = torch.cat([src_not_done, tar_not_done], 0)

        q_loss_step = self.update_q_functions(state, action, reward, next_state, not_done, writer)

        self.q_optimizer.zero_grad()
        q_loss_step.backward()
        self.q_optimizer.step()

        # Delayed policy and target updates
        if self.total_it % self.config.get('policy_freq', 2) == 0:
            # update policy and temperature parameter
            for p in self.q_funcs.parameters():
                p.requires_grad = False

            pi_loss_step, a_loss_step = self.update_policy_and_temp(state)
            self.policy_optimizer.zero_grad()
            pi_loss_step.backward()
            self.policy_optimizer.step()

            if self.config['temperature_opt']:
                self.temp_optimizer.zero_grad()
                a_loss_step.backward()
                self.temp_optimizer.step()

            for p in self.q_funcs.parameters():
                p.requires_grad = True

            self.update_target()


    @property
    def alpha(self):
        return self.log_alpha.exp()

    def save(self, filename):
        torch.save(self.q_funcs.state_dict(), filename + "_critic")
        torch.save(self.q_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.policy.state_dict(), filename + "_actor")
        torch.save(self.policy_optimizer.state_dict(), filename + "_actor_optimizer")

        torch.save(self.metric_model.state_dict(), filename + "_metric_model")
        torch.save(self.dynamics_optimizer.state_dict(), filename + "_dynamics_optimizer")


    def load(self, filename):
        self.q_funcs.load_state_dict(torch.load(filename + "_critic"))
        self.q_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.policy.load_state_dict(torch.load(filename + "_actor"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))

        self.metric_model.load_state_dict(torch.load(filename + "_metric_model"))
        self.dynamics_optimizer.load_state_dict(torch.load(filename + "_dynamics_optimizer"))
