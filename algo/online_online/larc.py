import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, constraints
from torch.distributions.transforms import Transform

# Import the models for LIBERTY exploration bonus
from liberty_models import MetricModel, InverseDynamicsModel, ForwardDynamicsModel


# --- Core SAC/DARC Components (Unaltered) ---

class TanhTransform(Transform):
    r"""Transform via the mapping :math:`y = \tanh(x)`."""
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


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_size=256):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.network = MLPNetwork(state_dim, action_dim * 2, hidden_size)

    def forward(self, state):
        log_std_min, log_std_max = -20, 2
        mean, log_std = self.network(state).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, log_std_min, log_std_max)
        std = log_std.exp()
        base_dist = Normal(mean, std)
        self.transform = TanhTransform(cache_size=1)
        return TransformedDistribution(base_dist, self.transform)

    def sample(self, state):
        dist = self.forward(state)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action * self.max_action, log_prob


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(QNetwork, self).__init__()
        self.q1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.q2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


class Classifier(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Classifier, self).__init__()
        self.network = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


# --- LARC Algorithm ---

class LARC(object):
    def __init__(self, config, device, target_entropy=None):
        # --- Standard Initialization (Identical to DARC) ---
        self.config = config
        self.device = device

        state_dim = config['state_dim']
        action_dim = config['action_dim']
        max_action = config['max_action']

        self.policy = Actor(state_dim, action_dim, max_action, config['hidden_size']).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['actor_lr'])
        self.q_funcs = QNetwork(state_dim, action_dim, config['hidden_size']).to(self.device)
        self.q_funcs_target = copy.deepcopy(self.q_funcs)
        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=config['critic_lr'])
        self.q_criterion = nn.MSELoss()

        self.classifier = Classifier(state_dim, action_dim, config['hidden_size']).to(self.device)
        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=config['actor_lr'])
        self.classifier_criterion = nn.BCEWithLogitsLoss()

        if target_entropy is None:
            self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(self.device)).item()
        else:
            self.target_entropy = target_entropy

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=config['actor_lr'])

        self.gamma = config['gamma']
        self.tau = config['tau']
        self._max_action = max_action

        # --- LARC-Specific State (for Lazy Initialization and Mode Switching) ---
        self._larc_models_initialized = False
        self.exploration_mode = False

    def _initialize_larc_models(self):
        """Initializes all models and optimizers specific to LARC on the first training call."""
        state_dim = self.config['state_dim']
        action_dim = self.config['action_dim']
        hidden_size = self.config['hidden_size']

        self.target_policy = Actor(state_dim, action_dim, self._max_action, hidden_size).to(self.device)
        self.target_policy_optimizer = torch.optim.Adam(self.target_policy.parameters(), lr=self.config['actor_lr'])
        self.target_q_funcs = QNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.target_q_funcs_target = copy.deepcopy(self.target_q_funcs)
        self.target_q_optimizer = torch.optim.Adam(self.target_q_funcs.parameters(), lr=self.config['critic_lr'])

        self.metric_model = MetricModel(state_dim, hidden_size).to(self.device)
        self.inverse_dynamics_model = InverseDynamicsModel(state_dim, action_dim, hidden_size).to(self.device)
        self.forward_dynamics_model = ForwardDynamicsModel(state_dim, action_dim, hidden_size).to(self.device)

        dynamics_params = list(self.metric_model.parameters()) + \
                          list(self.inverse_dynamics_model.parameters()) + \
                          list(self.forward_dynamics_model.parameters())
        self.dynamics_optimizer = torch.optim.Adam(dynamics_params, lr=self.config.get('liberty_lr', 3e-4))
        self.initial_state = None

        self.target_log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.target_temp_optimizer = torch.optim.Adam([self.target_log_alpha], lr=self.config['actor_lr'])

        self._larc_models_initialized = True

    def enable_exploration(self):
        self.exploration_mode = True

    def disable_exploration(self):
        self.exploration_mode = False

    def select_action(self, state):
        if self.exploration_mode:
            # The exploration policy is guaranteed to be initialized by the time this is called
            # because train() is called before the first step in the target environment.
            return self._select_target_action(state)
        else:
            return self._select_main_action(state)

    def _select_main_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        dist = self.policy(state)
        action = dist.sample() # using sample() from the Actor class
        return action.clamp(-1, 1).cpu().data.numpy().flatten()

    def _select_target_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        dist = self.target_policy(state)
        action = dist.sample()
        return action.clamp(-1, 1).cpu().data.numpy().flatten()

    def p_source(self, state, action):
        return torch.sigmoid(self.classifier(state, action / self._max_action))

    def train(self, src_replay_buffer, tar_replay_buffer, writer, iterations):
        if not self._larc_models_initialized:
            self._initialize_larc_models()

        for it in range(iterations):
            src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample(self.config['batch_size'])
            tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(self.config['batch_size'])

            self.update_classifier(src_state, src_action, tar_state, tar_action)
            self.update_liberty_models(tar_replay_buffer, self.config['batch_size'])
            self.update_target_policy(tar_state, tar_action, tar_next_state, tar_not_done)
            self.update_main_models(src_state, src_action, src_next_state, src_reward, src_not_done,
                                    tar_state, tar_action, tar_next_state, tar_not_done)

    def update_classifier(self, src_state, src_action, tar_state, tar_action):
        src_logits = self.classifier(src_state, src_action / self._max_action)
        tar_logits = self.classifier(tar_state, tar_action / self._max_action)
        src_labels = torch.ones_like(src_logits)
        tar_labels = torch.zeros_like(tar_logits)
        logits = torch.cat([src_logits, tar_logits], dim=0)
        labels = torch.cat([src_labels, tar_labels], dim=0)
        classifier_loss = self.classifier_criterion(logits, labels)
        self.classifier_optimizer.zero_grad()
        classifier_loss.backward()
        self.classifier_optimizer.step()

    def update_liberty_models(self, tar_replay_buffer, batch_size):
        s1, a1, ns1, _, _ = tar_replay_buffer.sample(batch_size)
        s2, a2, ns2, _, _ = tar_replay_buffer.sample(batch_size)
        pred_a1 = self.inverse_dynamics_model(s1, ns1)
        pred_a2 = self.inverse_dynamics_model(s2, ns2)
        inv_loss = F.mse_loss(pred_a1, a1) + F.mse_loss(pred_a2, a2)
        pred_ns1_mu, pred_ns1_log_std = self.forward_dynamics_model(s1, a1)
        dist1 = Normal(pred_ns1_mu, pred_ns1_log_std.exp())
        nll1 = -dist1.log_prob(ns1).mean()
        pred_ns2_mu, pred_ns2_log_std = self.forward_dynamics_model(s2, a2)
        dist2 = Normal(pred_ns2_mu, pred_ns2_log_std.exp())
        nll2 = -dist2.log_prob(ns2).mean()
        fwd_loss = nll1 + nll2
        with torch.no_grad():
            reward_dist = F.mse_loss(pred_ns1_mu, pred_ns2_mu, reduction='none').mean(dim=1)
            action_dist = F.mse_loss(pred_a1, pred_a2, reduction='none').mean(dim=1)
            metric_target = reward_dist + action_dist
        metric_pred = self.metric_model(s1, s2).squeeze()
        metric_loss = F.mse_loss(metric_pred, metric_target)
        total_loss = self.config.get('inv_loss_coeff', 1.0) * inv_loss + \
                     self.config.get('fwd_loss_coeff', 1.0) * fwd_loss + \
                     self.config.get('metric_loss_coeff', 1.0) * metric_loss
        self.dynamics_optimizer.zero_grad()
        total_loss.backward()
        self.dynamics_optimizer.step()

    def update_target_policy(self, state, action, next_state, not_done):
        with torch.no_grad():
            if self.initial_state is None:
                self.initial_state = state[0].unsqueeze(0).clone().detach()
            phi_s_prime = self.metric_model(next_state, self.initial_state)
            phi_s = self.metric_model(state, self.initial_state)
            intrinsic_reward = (self.gamma * phi_s_prime - phi_s).squeeze(-1)
        with torch.no_grad():
            next_action, next_log_pi = self.target_policy.sample(next_state)
            q1_next, q2_next = self.target_q_funcs_target(next_state, next_action)
            min_q_next = torch.min(q1_next, q2_next).squeeze(-1)
            target_v = min_q_next - self.target_alpha * next_log_pi
            q_target = intrinsic_reward + self.gamma * not_done * target_v
        q1, q2 = self.target_q_funcs(state, action)
        q_loss = self.q_criterion(q1.squeeze(-1), q_target) + self.q_criterion(q2.squeeze(-1), q_target)
        self.target_q_optimizer.zero_grad()
        q_loss.backward()
        self.target_q_optimizer.step()
        pi, log_pi = self.target_policy.sample(state)
        q1_pi, q2_pi = self.target_q_funcs(state, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.target_alpha.detach() * log_pi - min_q_pi).mean()
        self.target_policy_optimizer.zero_grad()
        policy_loss.backward()
        self.target_policy_optimizer.step()
        alpha_loss = -(self.target_log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.target_temp_optimizer.zero_grad()
        alpha_loss.backward()
        self.target_temp_optimizer.step()
        self.update_target(is_target_policy=True)

    def update_main_models(self, src_state, src_action, src_next_state, src_reward, src_not_done,
                                 tar_state, tar_action, tar_next_state, tar_not_done):
        with torch.no_grad():
            next_action, next_log_pi = self.policy.sample(src_next_state)
            tar_next_action, tar_next_log_pi = self.policy.sample(tar_next_state)
            q1_next_src, q2_next_src = self.q_funcs_target(src_next_state, next_action)
            min_q_next_src = torch.min(q1_next_src, q2_next_src).squeeze(-1)
            target_v_src = min_q_next_src - self.alpha * next_log_pi
            q_target_src = src_reward + self.gamma * src_not_done * target_v_src
            darc_reward = -torch.log(self.p_source(tar_state, tar_action) + 1e-8).squeeze(-1)
            if self.initial_state is None:
                self.initial_state = tar_state[0].unsqueeze(0).clone().detach()
            phi_s_prime = self.metric_model(tar_next_state, self.initial_state)
            phi_s = self.metric_model(tar_state, self.initial_state)
            liberty_reward = (self.gamma * phi_s_prime - phi_s).squeeze(-1)
            combined_reward = self.config.get(['darc_lambda'], 1.0) * darc_reward + \
                              self.config.get('liberty_lambda', 1.0) * liberty_reward
            q1_next_tar, q2_next_tar = self.q_funcs_target(tar_next_state, tar_next_action)
            min_q_next_tar = torch.min(q1_next_tar, q2_next_tar).squeeze(-1)
            target_v_tar = min_q_next_tar - self.alpha * tar_next_log_pi
            q_target_tar = combined_reward + self.gamma * tar_not_done * target_v_tar
        q1_src, q2_src = self.q_funcs(src_state, src_action)
        q1_tar, q2_tar = self.q_funcs(tar_state, tar_action)
        q_loss = self.q_criterion(q1_src.squeeze(-1), q_target_src) + self.q_criterion(q2_src.squeeze(-1), q_target_src) + \
                 self.q_criterion(q1_tar.squeeze(-1), q_target_tar) + self.q_criterion(q2_tar.squeeze(-1), q_target_tar)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        pi, log_pi = self.policy.sample(src_state)
        q1_pi, q2_pi = self.q_funcs(src_state, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha * log_pi - min_q_pi).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        if self.config['temperature_opt']:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.temp_optimizer.zero_grad()
            alpha_loss.backward()
            self.temp_optimizer.step()
        self.update_target(is_target_policy=False)

    def update_target(self, is_target_policy=False):
        if is_target_policy:
            q_funcs, q_funcs_target = self.target_q_funcs, self.target_q_funcs_target
        else:
            q_funcs, q_funcs_target = self.q_funcs, self.q_funcs_target
        for param, target_param in zip(q_funcs.parameters(), q_funcs_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def target_alpha(self):
        return self.target_log_alpha.exp()

    def save(self, filename):
        torch.save(self.q_funcs.state_dict(), filename + "_critic")
        torch.save(self.q_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.policy.state_dict(), filename + "_actor")
        torch.save(self.policy_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.classifier.state_dict(), filename + "_classifier")
        torch.save(self.classifier_optimizer.state_dict(), filename + "_classifier_optimizer")

        if self._larc_models_initialized:
            torch.save(self.target_q_funcs.state_dict(), filename + "_target_critic")
            torch.save(self.target_q_optimizer.state_dict(), filename + "_target_critic_optimizer")
            torch.save(self.target_policy.state_dict(), filename + "_target_actor")
            torch.save(self.target_policy_optimizer.state_dict(), filename + "_target_actor_optimizer")
            torch.save(self.metric_model.state_dict(), filename + "_metric_model")
            torch.save(self.inverse_dynamics_model.state_dict(), filename + "_inv_dyn_model")
            torch.save(self.forward_dynamics_model.state_dict(), filename + "_fwd_dyn_model")
            torch.save(self.dynamics_optimizer.state_dict(), filename + "_dynamics_optimizer")

    def load(self, filename):
        self.q_funcs.load_state_dict(torch.load(filename + "_critic"))
        self.q_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.policy.load_state_dict(torch.load(filename + "_actor"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.classifier.load_state_dict(torch.load(filename + "_classifier"))
        self.classifier_optimizer.load_state_dict(torch.load(filename + "_classifier_optimizer"))

        if not self._larc_models_initialized:
            self._initialize_larc_models()
        self.target_q_funcs.load_state_dict(torch.load(filename + "_target_critic"))
        self.target_q_optimizer.load_state_dict(torch.load(filename + "_target_critic_optimizer"))
        self.target_policy.load_state_dict(torch.load(filename + "_target_actor"))
        self.target_policy_optimizer.load_state_dict(torch.load(filename + "_target_actor_optimizer"))
        self.metric_model.load_state_dict(torch.load(filename + "_metric_model"))
        self.inverse_dynamics_model.load_state_dict(torch.load(filename + "_inv_dyn_model"))
        self.forward_dynamics_model.load_state_dict(torch.load(filename + "_fwd_dyn_model"))
        self.dynamics_optimizer.load_state_dict(torch.load(filename + "_dynamics_optimizer"))

