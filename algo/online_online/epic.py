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

    def get_logprob_of(self, state, action):
        mu_logstd = self.network(state)
        mu, logstd = mu_logstd.chunk(2, dim=1)
        logstd = torch.clamp(logstd, -20, 2)
        std = logstd.exp()
        dist = Normal(mu, std)
        transforms = [TanhTransform(cache_size=1)]
        dist = TransformedDistribution(dist, transforms)
        # Un-scale the action to be in the [-1, 1] range of the Tanh transform
        unscaled_action = action / self.max_action
        # Clamp to avoid issues with atanh
        unscaled_action = torch.clamp(unscaled_action, -0.99999, 0.99999)
        return dist.log_prob(unscaled_action).sum(axis=-1, keepdim=True)


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


class EPIC(object):

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

        # aka critic
        self.q_funcs = DoubleQFunc(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # aka actor (main source policy)
        self.policy = Policy(config['state_dim'], config['action_dim'], config['max_action'], hidden_size=config['hidden_sizes']).to(self.device)

        # New, separate policy for exploration in the target environment
        self.exploration_policy = Policy(config['state_dim'], config['action_dim'], config['max_action'], hidden_size=config['hidden_sizes']).to(self.device)
        self.exploration_mode = False  # Flag to switch between policies

        # aka temperature
        if config['temperature_opt']:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        else:
            self.log_alpha = torch.log(torch.FloatTensor([0.2])).to(self.device)

        # DARC classifier
        self.classifier = Classifier(config['state_dim'], config['action_dim'], config['hidden_sizes'], config['gaussian_noise_std']).to(self.device)

        # --- LIBERTY Exploration Components ---
        self.metric_model = MetricModel(config['state_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.inverse_model = InverseDynamicsModel(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.forward_model = ForwardDynamicsModel(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)

        self.metric_optimizer = torch.optim.Adam(self.metric_model.parameters(), lr=config.get('liberty_lr', 3e-4))
        self.dynamics_optimizer = torch.optim.Adam(
            list(self.inverse_model.parameters()) + list(self.forward_model.parameters()),
            lr=config.get('liberty_lr', 3e-4)
        )
        self.intrinsic_reward_coef = config.get('intrinsic_reward_coef', 0.003)
        # --- End LIBERTY Components ---

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=config['critic_lr'])
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['actor_lr'])
        exploration_actor_lr = config.get('exploration_actor_lr', config['actor_lr'])
        self.exploration_policy_optimizer = torch.optim.Adam(self.exploration_policy.parameters(), lr=exploration_actor_lr)
        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=config['actor_lr'])
        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=config['actor_lr'])

        self.use_importance_sampling = config.get('use_importance_sampling', True)
        self.use_dynamics_sampling = config.get('use_dynamics_sampling', True)

    def select_action(self, state, test=True):
        with torch.no_grad():
            if self.exploration_mode and not test:
                # Use exploration policy for training in the target environment
                policy_to_use = self.exploration_policy
            else:
                # Use the main policy for the source env and for all testing
                policy_to_use = self.policy

            action, _, mean = policy_to_use(torch.Tensor(state).view(1,-1).to(self.device))

        if test:
            return mean.squeeze().cpu().numpy()
        else:
            return action.squeeze().cpu().numpy()

    def enable_exploration(self):
        """Switches to the exploration policy for action selection."""
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

        # set labels for different domains
        label = torch.cat([torch.zeros(size=(batch_size,)), torch.ones(size=(batch_size,))], dim=0).long().to(self.device)

        indexs = torch.randperm(label.shape[0])
        state_batch, action_batch, nextstate_batch = state[indexs], action[indexs], next_state[indexs]
        label = label[indexs]

        sas_logits, sa_logits = self.classifier(state_batch, action_batch, nextstate_batch, with_noise=True)
        loss_sas = F.cross_entropy(sas_logits, label)
        loss_sa = F.cross_entropy(sa_logits, label)
        classifier_loss = loss_sas + loss_sa
        self.classifier_optimizer.zero_grad()
        classifier_loss.backward()
        self.classifier_optimizer.step()

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/sas classifier loss', loss_sas, global_step=self.total_it)
            writer.add_scalar('train/sa classifier loss', loss_sa, global_step=self.total_it)

    def _update_liberty_models(self, state, action, reward, next_state, writer):
        # Inverse dynamics loss
        pred_action = self.inverse_model(state, next_state)
        inverse_loss = F.mse_loss(pred_action, action)

        # Forward dynamics loss
        pred_next_state_mu, pred_next_state_std = self.forward_model(state, action)
        forward_loss = -Normal(pred_next_state_mu, pred_next_state_std).log_prob(next_state).mean()

        self.dynamics_optimizer.zero_grad()
        (inverse_loss + forward_loss).backward()
        self.dynamics_optimizer.step()

        # Metric model loss
        perm = torch.randperm(state.size(0))
        state_p, action_p, reward_p, next_state_p = state[perm], action[perm], reward[perm], next_state[perm]

        with torch.no_grad():
            r_dist = F.mse_loss(reward, reward_p, reduction='none')

            pred_mu, pred_std = self.forward_model(state, action)
            pred_mu_p, pred_std_p = self.forward_model(state_p, action_p)
            w2_dist_sq = F.mse_loss(pred_mu, pred_mu_p, reduction='none').sum(dim=-1, keepdim=True) + \
                         F.mse_loss(pred_std, pred_std_p, reduction='none').sum(dim=-1, keepdim=True)
            w2_dist = torch.sqrt(w2_dist_sq + 1e-6)

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

    def update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def update_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, not_done_batch, writer=None):
        with torch.no_grad():
            nextaction_batch, logprobs_batch, _ = self.policy(nextstate_batch, get_logprob=True)
            q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
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

    def update_exploration_policy(self, state_batch):
        """Updates the exploration policy using only the target state batch."""
        action_batch, logprobs_batch, _ = self.exploration_policy(state_batch, get_logprob=True)
        q_b1, q_b2 = self.q_funcs(state_batch, action_batch)
        qval_batch = torch.min(q_b1, q_b2)
        # We use detached alpha as temperature is optimized by the main policy
        policy_loss = (self.alpha.detach() * logprobs_batch - qval_batch).mean()
        return policy_loss

    def _get_dynamics_weights(self, target_states, target_actions, target_next_states):
        """Calculate dynamics importance sampling weights using the DARC classifier."""
        with torch.no_grad():
            sas_probs, sa_probs = self.classifier(target_states, target_actions, target_next_states, with_noise=False)
            sas_log_probs, sa_log_probs = torch.log(sas_probs + 1e-10), torch.log(sa_probs + 1e-10)
            
            # This is log( p_tar(s'|s,a) / p_src(s'|s,a) )
            reward_penalty = sas_log_probs[:, 1:] - sa_log_probs[:, 1:] - sas_log_probs[:, :1] + sa_log_probs[:, :1]
            
            # We want the inverse ratio: p_src / p_tar
            weights = torch.exp(-reward_penalty)
            
            # Clamp for stability
            weights = torch.clamp(weights, 0.0, 10.0)
        return weights

    def _get_importance_weights(self, target_states, target_actions):
        """Calculate importance sampling weights."""
        with torch.no_grad():
            # Probability of action under the main policy
            log_prob_main = self.policy.get_logprob_of(target_states, target_actions)
            # Probability of action under the exploration policy
            log_prob_exp = self.exploration_policy.get_logprob_of(target_states, target_actions)
            
            # Ratio: pi_main / pi_exploration
            weights = torch.exp(log_prob_main - log_prob_exp)
            
            # Clamp weights for stability
            weights = torch.clamp(weights, 0.0, 10.0)
        return weights

    def train(self, src_replay_buffer, tar_replay_buffer, initial_state, batch_size=128, writer=None):

        self.total_it += 1

        if src_replay_buffer.size < batch_size or tar_replay_buffer.size < batch_size:
            return

        # --- Update LIBERTY models on target data ---
        tar_state_lib, tar_action_lib, tar_next_state_lib, tar_reward_lib, _ = tar_replay_buffer.sample(batch_size)
        self._update_liberty_models(tar_state_lib, tar_action_lib, tar_reward_lib, tar_next_state_lib, writer)

        # --- DARC reward shaping for source data ---
        if self.total_it > self.config.get('darc_warmup_steps', 100000):
            if self.total_it % self.config['tar_env_interact_freq'] == 0:
                self.update_classifier(src_replay_buffer, tar_replay_buffer, batch_size, writer)

            src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample(batch_size)

            with torch.no_grad():
                sas_probs, sa_probs = self.classifier(src_state, src_action, src_next_state, with_noise=False)
                sas_log_probs, sa_log_probs = torch.log(sas_probs + 1e-10), torch.log(sa_probs + 1e-10)
                reward_penalty = sas_log_probs[:, 1:] - sa_log_probs[:, 1:] - sas_log_probs[:, :1] + sa_log_probs[:, :1]

                if writer is not None and self.total_it % 5000 == 0:
                    writer.add_scalar('train/reward penalty', reward_penalty.mean(), global_step=self.total_it)

                src_reward += self.config['penalty_coefficient'] * reward_penalty
        else:
            src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample(batch_size)

        # --- LIBERTY intrinsic reward for target data ---
        tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(batch_size)

        if initial_state is not None:
            with torch.no_grad():
                initial_state_batch = torch.Tensor(initial_state).expand_as(tar_state).to(self.device)
                potential_s = self.metric_model(tar_state, initial_state_batch)
                potential_s_next = self.metric_model(tar_next_state, initial_state_batch)
                intrinsic_reward = self.discount * potential_s_next - potential_s

                if writer is not None and self.total_it % 1000 == 0:
                    writer.add_scalar('liberty/intrinsic_reward_mean', intrinsic_reward.mean().item(), self.total_it)

                tar_reward += self.intrinsic_reward_coef * intrinsic_reward

        # Combine batches for SAC Q-function update
        state = torch.cat([src_state, tar_state], 0)
        action = torch.cat([src_action, tar_action], 0)
        next_state = torch.cat([src_next_state, tar_next_state], 0)
        reward = torch.cat([src_reward, tar_reward], 0)
        not_done = torch.cat([src_not_done, tar_not_done], 0)

        # --- Importance Sampling for Target Data ---
        # Policy IS weights
        if self.use_importance_sampling:
            w_policy = self._get_importance_weights(tar_state, tar_action)
        else:
            w_policy = torch.ones_like(tar_reward)

        # Dynamics IS weights
        if self.use_dynamics_sampling and self.total_it > self.config.get('darc_warmup_steps', 100000):
            w_dynamics = self._get_dynamics_weights(tar_state, tar_action, tar_next_state)
        else:
            w_dynamics = torch.ones_like(tar_reward)
        
        is_weights = w_policy * w_dynamics

        # --- Q-Function Update ---
        with torch.no_grad():
            nextaction_batch, logprobs_batch, _ = self.policy(next_state, get_logprob=True)
            q_t1, q_t2 = self.target_q_funcs(next_state, nextaction_batch)
            q_target = torch.min(q_t1, q_t2)
            value_target = reward + not_done * self.discount * (q_target - self.alpha * logprobs_batch)
        
        q_1, q_2 = self.q_funcs(state, action)
        
        # Separate losses for source and target
        q1_loss_src = F.mse_loss(q_1[:batch_size], value_target[:batch_size])
        q2_loss_src = F.mse_loss(q_2[:batch_size], value_target[:batch_size])
        
        # Apply IS weights to target loss
        q1_loss_tar = (is_weights * (q_1[batch_size:] - value_target[batch_size:])**2).mean()
        q2_loss_tar = (is_weights * (q_2[batch_size:] - value_target[batch_size:])**2).mean()

        q_loss_step = q1_loss_src + q2_loss_src + q1_loss_tar + q2_loss_tar

        self.q_optimizer.zero_grad()
        q_loss_step.backward()
        self.q_optimizer.step()

        # Delayed policy and target updates
        if self.total_it % self.config.get('policy_freq', 2) == 0:
            for p in self.q_funcs.parameters():
                p.requires_grad = False

            # --- Main Policy and Temperature Update ---
            action_batch, logprobs_batch, _ = self.policy(state, get_logprob=True)
            q_b1, q_b2 = self.q_funcs(state, action_batch)
            qval_batch = torch.min(q_b1, q_b2)

            # Separate the log-probs for source and target
            logprobs_src, logprobs_tar = logprobs_batch.chunk(2, dim=0)

            # Weight the target policy loss
            policy_loss_src = (self.alpha * logprobs_src - qval_batch[:batch_size]).mean()
            policy_loss_tar = (is_weights * (self.alpha * logprobs_tar - qval_batch[batch_size:])).mean()
            pi_loss_step = policy_loss_src + policy_loss_tar

            # Temperature loss is calculated on the full batch
            a_loss_step = -self.alpha * (logprobs_batch.detach() + self.target_entropy).mean()

            self.policy_optimizer.zero_grad()
            pi_loss_step.backward()
            self.policy_optimizer.step()

            if self.config['temperature_opt']:
                self.temp_optimizer.zero_grad()
                a_loss_step.backward()
                self.temp_optimizer.step()

            # --- Exploration Policy Update (UNCHANGED) ---
            # This update remains untouched, using only target data and no importance sampling.
            exp_pi_loss_step = self.update_exploration_policy(tar_state)
            self.exploration_policy_optimizer.zero_grad()
            exp_pi_loss_step.backward()
            self.exploration_policy_optimizer.step()

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

        torch.save(self.exploration_policy.state_dict(), filename + "_exploration_actor")
        torch.save(self.exploration_policy_optimizer.state_dict(), filename + "_exploration_actor_optimizer")

        torch.save(self.classifier.state_dict(), filename + "_classifier")
        torch.save(self.classifier_optimizer.state_dict(), filename + "_classifier_optimizer")

        torch.save(self.metric_model.state_dict(), filename + "_metric_model")
        torch.save(self.dynamics_optimizer.state_dict(), filename + "_dynamics_optimizer")

    def load(self, filename):
        self.q_funcs.load_state_dict(torch.load(filename + "_critic"))
        self.q_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))

        self.policy.load_state_dict(torch.load(filename + "_actor"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))

        self.exploration_policy.load_state_dict(torch.load(filename + "_exploration_actor"))
        self.exploration_policy_optimizer.load_state_dict(torch.load(filename + "_exploration_actor_optimizer"))

        self.classifier.load_state_dict(torch.load(filename + "_classifier"))
        self.classifier_optimizer.load_state_dict(torch.load(filename + "_classifier_optimizer"))

        self.metric_model.load_state_dict(torch.load(filename + "_metric_model"))
        self.dynamics_optimizer.load_state_dict(torch.load(filename + "_dynamics_optimizer"))
