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
        # The action from the buffer is already tanh-squashed and scaled.
        # We need to inverse the transformations to compute the log_prob.
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

# New Components for NOMAD-RND
class SourceForwardModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(SourceForwardModel, self).__init__()
        self.network = MLPNetwork(state_dim + action_dim, state_dim * 2, hidden_size)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        mu, log_std = self.network(x).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -5, 2)
        std = log_std.exp()
        return Normal(mu, std)

class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.count = epsilon

    def update(self, x):
        batch_mean, batch_var = torch.mean(x, dim=0), torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.pow(delta, 2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean.to(x.device)) / torch.sqrt(self.var.to(x.device) + 1e-8)


class NOMAD_RND(object):

    def __init__(self, config, device, target_entropy=None):
        self.config = config
        self.device = device
        self.discount = config['gamma']
        self.tau = config['tau']
        self.target_entropy = target_entropy if target_entropy else -config['action_dim']
        self.total_it = 0

        # Task policy components
        self.policy = Policy(config['state_dim'], config['action_dim'], config['max_action'], hidden_size=config['hidden_sizes']).to(self.device)
        self.task_q_funcs = DoubleQFunc(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.target_task_q_funcs = copy.deepcopy(self.task_q_funcs)
        self.target_task_q_funcs.eval()
        for p in self.target_task_q_funcs.parameters():
            p.requires_grad = False

        # Exploratory policy components
        self.exploration_policy = Policy(config['state_dim'], config['action_dim'], config['max_action'], hidden_size=config['hidden_sizes']).to(self.device)
        self.exp_q_funcs = DoubleQFunc(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.target_exp_q_funcs = copy.deepcopy(self.exp_q_funcs)
        self.target_exp_q_funcs.eval()
        for p in self.target_exp_q_funcs.parameters():
            p.requires_grad = False
        
        self.exploration_mode = False

        # Temperature (alpha) for SAC
        if config['temperature_opt']:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        else:
            self.log_alpha = torch.log(torch.FloatTensor([0.2])).to(self.device)

        # DARC domain classifier for dynamics importance weights
        self.classifier = Classifier(config['state_dim'], config['action_dim'], config['hidden_sizes'], config['gaussian_noise_std']).to(self.device)

        # --- New RND Components ---
        self.num_source_models = config.get('num_source_models', 7)
        self.source_models = [SourceForwardModel(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device) for _ in range(self.num_source_models)]
        self.source_models_optimizer = torch.optim.Adam(
            [p for model in self.source_models for p in model.parameters()],
            lr=config.get('source_model_lr', 3e-4)
        )
        self.state_normalizer = RunningMeanStd(shape=(config['state_dim'],))
        self.intrinsic_reward_coef = config.get('intrinsic_reward_coef', 0.003)
        self.model_update_freq = config.get('model_update_freq', 250)
        # --- End RND Components ---

        # Optimizers
        self.task_q_optimizer = torch.optim.Adam(self.task_q_funcs.parameters(), lr=config['critic_lr'])
        self.exp_q_optimizer = torch.optim.Adam(self.exp_q_funcs.parameters(), lr=config['critic_lr'])
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['actor_lr'])
        self.exploration_policy_optimizer = torch.optim.Adam(self.exploration_policy.parameters(), lr=config.get('exploration_actor_lr', config['actor_lr']))
        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=config['actor_lr'])
        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=config['actor_lr'])

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
        label = torch.cat([torch.zeros(size=(batch_size,)), torch.ones(size=(batch_size,))], dim=0).long().to(self.device)
        indices = torch.randperm(label.shape[0])
        state_batch, action_batch, nextstate_batch, label_batch = state[indices], action[indices], next_state[indices], label[indices]
        sas_logits, sa_logits = self.classifier(state_batch, action_batch, nextstate_batch, with_noise=True)
        loss = F.cross_entropy(sas_logits, label_batch) + F.cross_entropy(sa_logits, label_batch)
        self.classifier_optimizer.zero_grad()
        loss.backward()
        self.classifier_optimizer.step()
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/classifier_loss', loss.item(), global_step=self.total_it)

    def _update_source_models(self, src_replay_buffer, batch_size, writer):
        total_loss = 0
        for model in self.source_models:
            state, action, next_state, _, _ = src_replay_buffer.sample(batch_size)
            dist = model(state, action)
            loss = -dist.log_prob(next_state).mean()
            total_loss += loss
        
        self.source_models_optimizer.zero_grad()
        total_loss.backward()
        self.source_models_optimizer.step()

        if writer is not None and self.total_it % 1000 == 0:
            writer.add_scalar('train/source_model_loss', total_loss.item() / len(self.source_models), self.total_it)

    def _get_contrastive_reward(self, state, action, next_state):
        with torch.no_grad():
            # Update running stats for normalization
            self.state_normalizer.update(next_state)
            
            # Pick a random model from the ensemble
            model_idx = np.random.randint(0, len(self.source_models))
            model = self.source_models[model_idx]
            
            # Predict next state from source dynamics
            pred_dist = model(state, action)
            pred_next_state = pred_dist.sample()
            
            # Calculate normalized distance
            norm_next_state = self.state_normalizer.normalize(next_state)
            norm_pred_next_state = self.state_normalizer.normalize(pred_next_state)
            
            intrinsic_reward = torch.linalg.norm(norm_next_state - norm_pred_next_state, dim=-1, keepdim=True)
        return intrinsic_reward

    def soft_update_target_networks(self):
        # Soft update for task policy's critic
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_task_q_funcs.parameters(), self.task_q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)
        # Soft update for exploratory policy's critic
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_exp_q_funcs.parameters(), self.exp_q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def train(self, src_replay_buffer, tar_replay_buffer, batch_size=128, writer=None):
        self.total_it += 1
        if src_replay_buffer.size < batch_size or tar_replay_buffer.size < batch_size:
            return

        # --- 1. Auxiliary Model Updates ---
        if self.total_it % self.model_update_freq == 0:
            self._update_source_models(src_replay_buffer, batch_size, writer)

        if self.total_it > self.config.get('darc_warmup_steps', 10000):
            if self.total_it % self.config.get('classifier_update_freq', 10) == 0:
                self.update_classifier(src_replay_buffer, tar_replay_buffer, batch_size, writer)

        # --- 2. Exploratory Policy Training Step ---
        tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(batch_size)

        with torch.no_grad():
            intrinsic_reward = self._get_contrastive_reward(tar_state, tar_action, tar_next_state)
            exp_reward = tar_reward + self.intrinsic_reward_coef * intrinsic_reward
            
            next_action_tar, log_prob_tar, _ = self.exploration_policy(tar_next_state, get_logprob=True)
            q_t1, q_t2 = self.target_exp_q_funcs(tar_next_state, next_action_tar)
            q_target = torch.min(q_t1, q_t2)
            value_target_exp = exp_reward + tar_not_done * self.discount * (q_target - self.alpha.detach() * log_prob_tar)

        q1_exp, q2_exp = self.exp_q_funcs(tar_state, tar_action)
        q_loss_exp = F.mse_loss(q1_exp, value_target_exp) + F.mse_loss(q2_exp, value_target_exp)

        self.exp_q_optimizer.zero_grad()
        q_loss_exp.backward()
        self.exp_q_optimizer.step()

        if self.total_it % self.config.get('policy_freq', 2) == 0:
            for p in self.exp_q_funcs.parameters(): p.requires_grad = False
            pi_action_tar, log_prob_pi_tar, _ = self.exploration_policy(tar_state, get_logprob=True)
            q1_pi, q2_pi = self.exp_q_funcs(tar_state, pi_action_tar)
            policy_loss_exp = (self.alpha.detach() * log_prob_pi_tar - torch.min(q1_pi, q2_pi)).mean()
            self.exploration_policy_optimizer.zero_grad()
            policy_loss_exp.backward()
            self.exploration_policy_optimizer.step()
            for p in self.exp_q_funcs.parameters(): p.requires_grad = True

        # --- 3. Task Policy Training Step ---
        src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample(batch_size)
        tar_state_is, tar_action_is, _, tar_reward_is, _ = tar_replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Q-target for source data
            next_action_src, log_prob_src, _ = self.policy(src_next_state, get_logprob=True)
            q_t1_src, q_t2_src = self.target_task_q_funcs(src_next_state, next_action_src)
            value_target_src = src_reward + src_not_done * self.discount * (torch.min(q_t1_src, q_t2_src) - self.alpha * log_prob_src)

            # Q-target for target data (using extrinsic reward)
            next_action_tar_is, log_prob_tar_is, _ = self.policy(tar_state_is, get_logprob=True)
            q_t1_tar_is, q_t2_tar_is = self.target_task_q_funcs(tar_state_is, next_action_tar_is)
            value_target_tar_is = tar_reward_is + tar_not_done * self.discount * (torch.min(q_t1_tar_is, q_t2_tar_is) - self.alpha * log_prob_tar_is)

            # Dynamics IS Weights for Source Data
            sas_probs_src, sa_probs_src = self.classifier(src_state, src_action, src_next_state, with_noise=False)
            darc_log_ratio_src = torch.log(sas_probs_src[:, 1:] + 1e-8) - torch.log(sas_probs_src[:, :1] + 1e-8) - \
                                 (torch.log(sa_probs_src[:, 1:] + 1e-8) - torch.log(sa_probs_src[:, :1] + 1e-8))
            w_dynamics = torch.exp(darc_log_ratio_src).detach()
            w_dynamics = torch.clamp(w_dynamics, 1e-4, 100.0)

            # Policy IS Weights for Target Data
            log_prob_task_policy = self.policy.log_prob(tar_state_is, tar_action_is)
            log_prob_exp_policy = self.exploration_policy.log_prob(tar_state_is, tar_action_is)
            w_policy = torch.exp(log_prob_task_policy - log_prob_exp_policy).detach()
            w_policy = torch.clamp(w_policy, 1e-4, 100.0)

        # Weighted Q-Loss for task policy
        q1_src, q2_src = self.task_q_funcs(src_state, src_action)
        q_loss_src = (w_dynamics * (F.mse_loss(q1_src, value_target_src, reduction='none') + F.mse_loss(q2_src, value_target_src, reduction='none'))).mean()
        q1_tar_is, q2_tar_is = self.task_q_funcs(tar_state_is, tar_action_is)
        q_loss_tar_is = (w_policy * (F.mse_loss(q1_tar_is, value_target_tar_is, reduction='none') + F.mse_loss(q2_tar_is, value_target_tar_is, reduction='none'))).mean()
        total_q_loss_task = q_loss_src + q_loss_tar_is
        self.task_q_optimizer.zero_grad()
        total_q_loss_task.backward()
        self.task_q_optimizer.step()

        # Task policy and temperature update (delayed)
        if self.total_it % self.config.get('policy_freq', 2) == 0:
            for p in self.task_q_funcs.parameters(): p.requires_grad = False
            
            pi_action_src, log_prob_pi_src, _ = self.policy(src_state, get_logprob=True)
            q1_pi_src, q2_pi_src = self.task_q_funcs(src_state, pi_action_src)
            policy_loss_src = (w_dynamics * (self.alpha * log_prob_pi_src - torch.min(q1_pi_src, q2_pi_src))).mean()

            pi_action_tar_is, log_prob_pi_tar_is, _ = self.policy(tar_state_is, get_logprob=True)
            q1_pi_tar_is, q2_pi_tar_is = self.task_q_funcs(tar_state_is, pi_action_tar_is)
            policy_loss_tar_is = (w_policy * (self.alpha * log_prob_pi_tar_is - torch.min(q1_pi_tar_is, q2_pi_tar_is))).mean()
            
            total_policy_loss = policy_loss_src + policy_loss_tar_is
            
            temp_loss = -self.alpha * (log_prob_pi_src.detach() + self.target_entropy).mean()

            self.policy_optimizer.zero_grad()
            total_policy_loss.backward()
            self.policy_optimizer.step()

            if self.config['temperature_opt']:
                self.temp_optimizer.zero_grad()
                temp_loss.backward()
                self.temp_optimizer.step()

            for p in self.task_q_funcs.parameters(): p.requires_grad = True
            
            self.soft_update_target_networks()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def save(self, filename):
        torch.save(self.policy.state_dict(), filename + "_task_actor")
        torch.save(self.exploration_policy.state_dict(), filename + "_exp_actor")
        torch.save(self.task_q_funcs.state_dict(), filename + "_task_critic")
        torch.save(self.exp_q_funcs.state_dict(), filename + "_exp_critic")
        torch.save(self.classifier.state_dict(), filename + "_classifier")
        # Save source models
        for i, model in enumerate(self.source_models):
            torch.save(model.state_dict(), f"{filename}_source_model_{i}")

    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename + "_task_actor"))
        self.exploration_policy.load_state_dict(torch.load(filename + "_exp_actor"))
        self.task_q_funcs.load_state_dict(torch.load(filename + "_task_critic"))
        self.exp_q_funcs.load_state_dict(torch.load(filename + "_exp_critic"))
        self.classifier.load_state_dict(torch.load(filename + "_classifier"))
        # Load source models
        for i, model in enumerate(self.source_models):
            model.load_state_dict(torch.load(f"{filename}_source_model_{i}"))
        
        self.target_task_q_funcs = copy.deepcopy(self.task_q_funcs)
        self.target_exp_q_funcs = copy.deepcopy(self.exp_q_funcs)
