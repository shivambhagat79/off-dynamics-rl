import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, constraints
from torch.distributions.transforms import Transform

# Assume liberty_models.py exists and import the necessary models
from liberty_models import MetricModel, InverseDynamicsModel, ForwardDynamicsModel

# The following helper classes (TanhTransform, MLPNetwork, SquashedNormal,
# GeneralPolicy, QNetwork) are reused from darc.py to maintain structure and compatibility.

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


class SquashedNormal(TransformedDistribution):
    r"""
    Creates a squashed Normal distribution by transforming a Normal distribution with a TanhTransform.
    """
    def __init__(self, loc, scale, validate_args=None):
        self.loc = loc
        self.scale = scale
        self.base_dist = Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms, validate_args=validate_args)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class GeneralPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(GeneralPolicy, self).__init__()
        self.action_dim = action_dim
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * action_dim)
        )

    def forward(self, state, get_logprob=False):
        mu_logstd = self.network(state)
        mu, log_std = mu_logstd.chunk(2, dim=1)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        dist = SquashedNormal(mu, std)

        action = dist.rsample()
        if get_logprob:
            log_prob = dist.log_prob(action).sum(axis=-1, keepdim=True)
            return action, log_prob
        return action


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(QNetwork, self).__init__()
        self.network1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.network2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.network1(x), self.network2(x)

    def min_q(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

class LARC(object):
    def __init__(self, config, device, target_entropy=None):
        self.config = config
        self.device = device
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.max_action = config['max_action']

        # --- Main DARC Policy (Source Policy) ---
        self.policy = GeneralPolicy(self.state_dim, self.action_dim, config['hidden_size']).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['policy_lr'])

        # --- LIBERTY Exploration Policy (Target Policy) ---
        self.target_policy = GeneralPolicy(self.state_dim, self.action_dim, config['hidden_size']).to(self.device)
        self.target_policy_optimizer = torch.optim.Adam(self.target_policy.parameters(), lr=config.get('liberty_policy_lr', 3e-4))
        self.exploration_enabled = False # Flag to switch between policies

        # --- Q-Networks and Classifier (Shared) ---
        self.q_funcs = QNetwork(self.state_dim, self.action_dim, config['hidden_size']).to(self.device)
        self.q_target_funcs = copy.deepcopy(self.q_funcs)
        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=config['q_lr'])

        self.classifier = MLPNetwork(self.state_dim * 2 + self.action_dim, 1, hidden_size=config['hidden_size']).to(self.device)
        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=config['cls_lr'])
        self.bce = torch.nn.BCEWithLogitsLoss()

        # --- LIBERTY Intrinsic Reward Models ---
        self.metric_model = MetricModel(self.state_dim, config['hidden_size']).to(self.device)
        self.inverse_model = InverseDynamicsModel(self.state_dim, self.action_dim, config['hidden_size']).to(self.device)
        self.forward_model = ForwardDynamicsModel(self.state_dim, self.action_dim, config['hidden_size']).to(self.device)

        self.dynamics_optimizer = torch.optim.Adam(
            list(self.metric_model.parameters()) +
            list(self.inverse_model.parameters()) +
            list(self.forward_model.parameters()),
            lr=config.get('liberty_lr', 3e-4)
        )
        self.initial_state = None # To be set on the first training call

        # --- Temperature for SAC ---
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config['policy_lr'])

        if target_entropy is None:
            self.target_entropy = -torch.prod(torch.Tensor(self.action_dim).to(self.device)).item()
        else:
            self.target_entropy = target_entropy

    def select_action(self, state, test=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            if test:
                # During evaluation, always use the main (source) policy and return the
                # deterministic mean of the action distribution for stable evaluation.
                mu_logstd = self.policy.network(state)
                mu, _ = mu_logstd.chunk(2, dim=1)
                action = torch.tanh(mu) # The mean of a SquashedNormal is tanh(mu)
            else:
                # During training/data collection, sample actions and switch
                # between policies based on the exploration flag.
                if self.exploration_enabled:
                    # Use target policy for exploration in the target domain
                    action = self.target_policy(state)
                else:
                    # Use source policy for interaction in the source domain
                    action = self.policy(state)
        return action.cpu().data.numpy().flatten()

    def enable_exploration(self):
        """Switches to the target policy for exploration."""
        self.exploration_enabled = True

    def disable_exploration(self):
        """Switches back to the source policy for evaluation."""
        self.exploration_enabled = False

    def train(self, src_replay_buffer, tar_replay_buffer, p_ratio, batch_size=256, writer=None):
        if self.initial_state is None:
            # Set the initial state for potential calculation from the first batch
            s, _, _, _, _ = tar_replay_buffer.sample(1)
            # FIX: 's' is already a tensor on the correct device from the replay buffer.
            # We just need to detach it from the computation graph.
            self.initial_state = s.detach()

        # Sample from both buffers
        src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample(batch_size)
        tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(batch_size)

        # 1. Update Classifier
        self.update_classifier(src_state, src_action, src_next_state, tar_state, tar_action, tar_next_state)

        # 2. Update LIBERTY models and Target Policy
        self.update_liberty_and_target_policy(tar_state, tar_action, tar_next_state)

        # 3. Update Q-Functions with DARC reward (source) and combined reward (target)
        self.update_q_functions(src_state, src_action, src_next_state, src_reward, src_not_done,
                                tar_state, tar_action, tar_next_state, tar_reward, tar_not_done)

        # 4. Update Source Policy and Temperature
        self.update_source_policy_and_temp(src_state)

        # 5. Update Target Networks
        self.update_target()

    def update_classifier(self, src_state, src_action, src_next_state, tar_state, tar_action, tar_next_state):
        src_trans = torch.cat([src_state, src_action, src_next_state], dim=1)
        tar_trans = torch.cat([tar_state, tar_action, tar_next_state], dim=1)

        src_logits = self.classifier(src_trans)
        tar_logits = self.classifier(tar_trans)

        # 1 for source, 0 for target
        src_loss = self.bce(src_logits, torch.ones_like(src_logits))
        tar_loss = self.bce(tar_logits, torch.zeros_like(tar_logits))

        classifier_loss = src_loss + tar_loss
        self.classifier_optimizer.zero_grad()
        classifier_loss.backward()
        self.classifier_optimizer.step()

    def update_liberty_and_target_policy(self, state, action, next_state):
        # --- 1. Update Dynamics and Metric Models (LIBERTY core) ---
        # Inverse dynamics loss
        pred_action = self.inverse_model(state, next_state)
        inverse_loss = F.mse_loss(pred_action, action)

        # Forward dynamics loss
        # FIX: The forward_model already returns a tuple (mu, logstd).
        # We should unpack it directly instead of calling .chunk().
        pred_next_state_mu, pred_next_state_logstd = self.forward_model(state, action)
        pred_next_state_std = torch.exp(pred_next_state_logstd)
        # Negative log-likelihood of the actual next_state
        forward_loss = -Normal(pred_next_state_mu, pred_next_state_std).log_prob(next_state).sum(axis=-1).mean()

        # Metric loss (bisimulation)
        dist_s_next_s = self.metric_model(state, next_state)
        dist_s_pred_next_s = self.metric_model(state, pred_next_state_mu.detach())
        metric_loss = F.mse_loss(dist_s_next_s, dist_s_pred_next_s)

        dynamics_loss = (self.config.get('liberty_forward_w', 0.2) * forward_loss +
                         self.config.get('liberty_inverse_w', 0.8) * inverse_loss +
                         self.config.get('liberty_metric_w', 1.0) * metric_loss)

        self.dynamics_optimizer.zero_grad()
        dynamics_loss.backward()
        self.dynamics_optimizer.step()

        # --- 2. Update Target (Exploration) Policy ---
        for p in self.q_funcs.parameters():
            p.requires_grad = False

        new_action, log_prob = self.target_policy(state, get_logprob=True)
        q_val = self.q_funcs.min_q(state, new_action)

        policy_loss = (self.alpha * log_prob - q_val).mean()

        self.target_policy_optimizer.zero_grad()
        policy_loss.backward()
        self.target_policy_optimizer.step()

        for p in self.q_funcs.parameters():
            p.requires_grad = True

    def update_q_functions(self, src_state, src_action, src_next_state, src_reward, src_not_done,
                           tar_state, tar_action, tar_next_state, tar_reward, tar_not_done):
        # --- Calculate DARC reward for source transitions ---
        with torch.no_grad():
            src_trans = torch.cat([src_state, src_action, src_next_state], dim=1)
            darc_reward = self.classifier(src_trans)
            augmented_src_reward = src_reward + self.config['darc_lambda'] * darc_reward

        # --- Calculate LIBERTY intrinsic reward for target transitions ---
        with torch.no_grad():
            # FIX: Expand self.initial_state to match the batch size of tar_state.
            batch_size = tar_state.shape[0]
            initial_state_batch = self.initial_state.expand(batch_size, -1)

            phi_s = self.metric_model(tar_state, initial_state_batch)
            phi_s_next = self.metric_model(tar_next_state, initial_state_batch)
            intrinsic_reward = self.config['gamma'] * phi_s_next - phi_s
            augmented_tar_reward = tar_reward + self.config.get('liberty_eta', 0.1) * intrinsic_reward

        # --- Compute target Q for both batches ---
        with torch.no_grad():
            # Use source policy for source transitions
            src_next_action, src_log_prob = self.policy(src_next_state, get_logprob=True)
            src_target_q = self.q_target_funcs.min_q(src_next_state, src_next_action)
            src_target_q = augmented_src_reward + src_not_done * self.config['gamma'] * (src_target_q - self.alpha * src_log_prob)

            # Use target policy for target transitions
            tar_next_action, tar_log_prob = self.target_policy(tar_next_state, get_logprob=True)
            tar_target_q = self.q_target_funcs.min_q(tar_next_state, tar_next_action)
            tar_target_q = augmented_tar_reward + tar_not_done * self.config['gamma'] * (tar_target_q - self.alpha * tar_log_prob)

        # --- Compute Q-loss ---
        src_q1, src_q2 = self.q_funcs(src_state, src_action)
        src_q_loss = F.mse_loss(src_q1, src_target_q) + F.mse_loss(src_q2, src_target_q)
        tar_q1, tar_q2 = self.q_funcs(tar_state, tar_action)
        tar_q_loss = F.mse_loss(tar_q1, tar_target_q) + F.mse_loss(tar_q2, tar_target_q)

        total_q_loss = src_q_loss + tar_q_loss

        self.q_optimizer.zero_grad()
        total_q_loss.backward()
        self.q_optimizer.step()

    def update_source_policy_and_temp(self, src_state):
        for p in self.q_funcs.parameters():
            p.requires_grad = False

        new_action, log_prob = self.policy(src_state, get_logprob=True)
        q_val = self.q_funcs.min_q(src_state, new_action)

        policy_loss = (self.alpha * log_prob - q_val).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update temperature
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for p in self.q_funcs.parameters():
            p.requires_grad = True

    def update_target(self):
        for param, target_param in zip(self.q_funcs.parameters(), self.q_target_funcs.parameters()):
            target_param.data.copy_(self.config['tau'] * param.data + (1 - self.config['tau']) * target_param.data)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def save(self, filename):
        torch.save(self.q_funcs.state_dict(), filename + "_critic")
        torch.save(self.q_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.policy.state_dict(), filename + "_actor")
        torch.save(self.policy_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.target_policy.state_dict(), filename + "_target_actor")
        torch.save(self.target_policy_optimizer.state_dict(), filename + "_target_actor_optimizer")
        torch.save(self.classifier.state_dict(), filename + "_classifier")
        torch.save(self.classifier_optimizer.state_dict(), filename + "_classifier_optimizer")
        # Save LIBERTY models
        torch.save(self.metric_model.state_dict(), filename + "_metric_model")
        torch.save(self.inverse_model.state_dict(), filename + "_inverse_model")
        torch.save(self.forward_model.state_dict(), filename + "_forward_model")
        torch.save(self.dynamics_optimizer.state_dict(), filename + "_dynamics_optimizer")

    def load(self, filename):
        self.q_funcs.load_state_dict(torch.load(filename + "_critic"))
        self.q_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.q_target_funcs = copy.deepcopy(self.q_funcs)
        self.policy.load_state_dict(torch.load(filename + "_actor"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.target_policy.load_state_dict(torch.load(filename + "_target_actor"))
        self.target_policy_optimizer.load_state_dict(torch.load(filename + "_target_actor_optimizer"))
        self.classifier.load_state_dict(torch.load(filename + "_classifier"))
        self.classifier_optimizer.load_state_dict(torch.load(filename + "_classifier_optimizer"))
        # Load LIBERTY models
        self.metric_model.load_state_dict(torch.load(filename + "_metric_model"))
        self.inverse_model.load_state_dict(torch.load(filename + "_inverse_model"))
        self.forward_model.load_state_dict(torch.load(filename + "_forward_model"))
        self.dynamics_optimizer.load_state_dict(torch.load(filename + "_dynamics_optimizer"))

class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )
    def forward(self, x):
        return self.network(x)




