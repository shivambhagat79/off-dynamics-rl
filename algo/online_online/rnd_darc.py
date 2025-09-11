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

# domain classifier for DARC
class Classifier(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_size=256, gaussian_noise_std=1.0):
        super(Classifier, self).__init__()
        self.action_dim = action_dim
        self.gaussian_noise_std = gaussian_noise_std
        self.sa_classifier = MLPNetwork(state_dim + action_dim, 2, hidden_size)
        self.sas_classifier = MLPNetwork(2*state_dim + action_dim, 2, hidden_size)

    def forward(self, state_batch, action_batch, nextstate_batch, with_noise):
        sas = torch.cat([state_batch, action_batch, nextstate_batch], -1)

        if with_noise:
            sas += torch.randn_like(sas, device=state_batch.device) * self.gaussian_noise_std
        sas_logits = torch.nn.Softmax()(self.sas_classifier(sas))

        sa = torch.cat([state_batch, action_batch], -1)

        if with_noise:
            sa += torch.randn_like(sa, device=state_batch.device) * self.gaussian_noise_std
        sa_logits = torch.nn.Softmax()(self.sa_classifier(sa))

        return sas_logits, sa_logits


class DARC(object):

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
        self.rnd_src_steps_per_iter = config['rnd_src_steps_per_iter']
        self.rnd_batch_size = config['rnd_batch_size']

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
        
        # aka classifier
        self.classifier = Classifier(config['state_dim'], config['action_dim'], config['hidden_sizes'], config['gaussian_noise_std']).to(self.device)

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=config['critic_lr'])
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['actor_lr'])
        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=config['actor_lr'])
        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=config['actor_lr'])

        # === Explorer SAC (target-only) ===
        self.exp_q_funcs = DoubleQFunc(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.exp_target_q_funcs = copy.deepcopy(self.exp_q_funcs)
        for p in self.exp_target_q_funcs.parameters():
            p.requires_grad = False

        self.exp_policy = Policy(config['state_dim'], config['action_dim'], config['max_action'], hidden_size=config['hidden_sizes']).to(self.device)

        # Separate temperature for explorer
        if config.get('exp_temperature_opt', True):
            self.exp_log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        else:
            self.exp_log_alpha = torch.log(torch.FloatTensor([0.2])).to(self.device)

        # Optimizers for explorer
        self.exp_q_optimizer = torch.optim.Adam(self.exp_q_funcs.parameters(), lr=config['critic_lr'])
        self.exp_policy_optimizer = torch.optim.Adam(self.exp_policy.parameters(), lr=config['actor_lr'])
        self.exp_temp_optimizer = torch.optim.Adam([self.exp_log_alpha], lr=config['actor_lr'])

        # === Forward-RND (source-aware) ===
        rnd_hidden = config.get('rnd_hidden', 256)
        self.rnd_target = MLPNetwork(config['state_dim'], rnd_hidden, hidden_size=rnd_hidden).to(self.device)
        for p in self.rnd_target.parameters():
            p.requires_grad = False  # fixed random target

        self.rnd_predictor = MLPNetwork(config['state_dim'] + config['action_dim'], rnd_hidden, hidden_size=rnd_hidden).to(self.device)

        self.rnd_opt = torch.optim.Adam(self.rnd_predictor.parameters(), lr=config.get('rnd_lr', 1e-3))
        self.rnd_running_mean = torch.zeros(1, device=self.device)
        self.rnd_running_var  = torch.ones(1, device=self.device)
        self.rnd_mom = 0.99

        # scale for intrinsic in explorer reward
        self.exp_beta = float(config.get('exp_beta', 0.1))
    
    @property
    def _exp_alpha(self):
     return self.exp_log_alpha.exp()

    def _update_target_net(self, target, source):
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    
    def select_action(self, state, test=True):
        with torch.no_grad():
            action, _, mean = self.policy(torch.Tensor(state).view(1,-1).to(self.device))
        if test:
            return mean.squeeze().cpu().numpy()
        else:
            return action.squeeze().cpu().numpy()
    
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
        loss_sa =  F.cross_entropy(sa_logits, label)
        classifier_loss = loss_sas + loss_sa
        self.classifier_optimizer.zero_grad()
        classifier_loss.backward()
        self.classifier_optimizer.step()

        # log necessary information if the logger is not None
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/sas classifier loss', loss_sas, global_step=self.total_it)
            writer.add_scalar('train/sa classifier loss', loss_sa, global_step=self.total_it)
    

    ## rnd training on source samples

    def prefit_rnd_on_source(self, src_replay_buffer, steps=10000, batch_size=256):
        self.rnd_predictor.train()
        for _ in range(steps):
            s, a, s2, _, _ = src_replay_buffer.sample(batch_size)
            with torch.no_grad():
                z_t1 = self.rnd_target(s2)  # random embedding of next state
            pred = self.rnd_predictor(torch.cat([s, a], dim=1))
            loss = F.mse_loss(pred, z_t1)
            self.rnd_opt.zero_grad()
            loss.backward()
            self.rnd_opt.step()

    


    @torch.no_grad()
    def _intrinsic_bonus(self, s, a, s2):
        """
        RND error normalized & clipped. Higher = more unlike source.
        """
        z_t1 = self.rnd_target(s2)
        pred = self.rnd_predictor(torch.cat([s, a], dim=1))
        err = (pred - z_t1).pow(2).mean(dim=1, keepdim=True)  # [B,1]

        # running normalize
        m = err.mean()
        v = err.var(unbiased=False) + 1e-8
        self.rnd_running_mean = self.rnd_mom * self.rnd_running_mean + (1 - self.rnd_mom) * m
        self.rnd_running_var  = self.rnd_mom * self.rnd_running_var  + (1 - self.rnd_mom) * v
        normed = (err - self.rnd_running_mean) / torch.sqrt(self.rnd_running_var + 1e-8)
        return normed.clamp(min=0.0, max=5.0)

    @torch.no_grad()
    def select_action_explorer(self, state, test=False):
        action, _, mean = self.exp_policy(torch.Tensor(state).view(1,-1).to(self.device))
        return (mean if test else action).squeeze().cpu().numpy()

    
    def explorer_update(self, tar_replay_buffer, batch_size=256, writer=None):
        if tar_replay_buffer.size < batch_size:
            return

        s, a, s2, r_env, not_done = tar_replay_buffer.sample(batch_size)

        # Intrinsic bonus (vs-source) and mixed reward for explorer
        r_int = self._intrinsic_bonus(s, a, s2)
        r = r_env + self.exp_beta * r_int

        # --- Q update ---
        with torch.no_grad():
            an, logp, _ = self.exp_policy(s2, get_logprob=True)
            q1t, q2t = self.exp_target_q_funcs(s2, an)
            q_t = torch.min(q1t, q2t)
            target = r + not_done * self.discount * (q_t - self._exp_alpha * logp)

        q1, q2 = self.exp_q_funcs(s, a)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.exp_q_optimizer.zero_grad()
        q_loss.backward()
        self.exp_q_optimizer.step()

        # soft update explorer target critics
        self._update_target_net(self.exp_target_q_funcs, self.exp_q_funcs)

        # --- policy + temperature updates ---
        for p in self.exp_q_funcs.parameters(): p.requires_grad = False
        ap, logp, _ = self.exp_policy(s, get_logprob=True)
        q1b, q2b = self.exp_q_funcs(s, ap)
        qv = torch.min(q1b, q2b)
        pi_loss = (self._exp_alpha * logp - qv).mean()
        self.exp_policy_optimizer.zero_grad()
        pi_loss.backward()
        self.exp_policy_optimizer.step()
        for p in self.exp_q_funcs.parameters(): p.requires_grad = True

        if self.config.get('exp_temperature_opt', True):
            temp_loss = -self._exp_alpha * (logp.detach() + (-self.config['action_dim'])).mean()
            self.exp_temp_optimizer.zero_grad()
            temp_loss.backward()
            self.exp_temp_optimizer.step()

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('explorer/q_loss', q_loss, self.total_it)
            writer.add_scalar('explorer/pi_loss', pi_loss, self.total_it)
            writer.add_scalar('explorer/temp_loss', temp_loss, self.total_it)

    def rnd_src_update(self, src_replay_buffer, iters=1, batch_size=256):
        """
        Keep training the RND predictor on SOURCE ONLY.
        Call this as often as you like (e.g., every train step).
        """
        self.rnd_predictor.train()
        for _ in range(int(iters)):
            s, a, s2, _, _ = src_replay_buffer.sample(batch_size)
            with torch.no_grad():
                z_t1 = self.rnd_target(s2)  # random embedding of next state
            pred = self.rnd_predictor(torch.cat([s, a], dim=1))
            loss = F.mse_loss(pred, z_t1)
            self.rnd_opt.zero_grad()
            loss.backward()
            self.rnd_opt.step()



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
            # value_target = reward_batch + not_done_batch * self.discount * q_target
        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/q1', q_1.mean(), self.total_it)
            writer.add_scalar('train/logprob', logprobs_batch.mean(), self.total_it)
        loss = F.mse_loss(q_1, value_target) + F.mse_loss(q_2, value_target)
        return loss, q_target

    def update_policy_and_temp(self, state_batch):
        action_batch, logprobs_batch, _ = self.policy(state_batch, get_logprob=True)
        q_b1, q_b2 = self.q_funcs(state_batch, action_batch)
        qval_batch = torch.min(q_b1, q_b2)
        policy_loss = (self.alpha * logprobs_batch - qval_batch).mean()
        temp_loss = -self.alpha * (logprobs_batch.detach() + self.target_entropy).mean()
        return policy_loss, temp_loss

    def train(self, src_replay_buffer, tar_replay_buffer, batch_size=128, writer=None):

        self.total_it += 1

        if src_replay_buffer.size < 2*batch_size or tar_replay_buffer.size < batch_size:
            return
        
        # follow the original paper, DARC has a warmup phase that does not involve reward modification
        if self.total_it <= int(1e5):
            src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample(2*batch_size)
            if self.config.get('rnd_src_steps_per_iter', 1) > 0:
                self.rnd_src_update(src_replay_buffer,
                            iters=self.config['rnd_src_steps_per_iter'],
                            batch_size=self.config.get('rnd_batch_size', batch_size))
        else:
            if self.total_it % self.config['tar_env_interact_freq'] == 0:
                self.update_classifier(src_replay_buffer, tar_replay_buffer, batch_size, writer)

            src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample(2 * batch_size)

            # we do reward modification
            with torch.no_grad():
                sas_probs, sa_probs = self.classifier(src_state, src_action, src_next_state, with_noise=False)
                sas_log_probs, sa_log_probs = torch.log(sas_probs + 1e-10), torch.log(sa_probs + 1e-10)
                reward_penalty = sas_log_probs[:, 1:] - sa_log_probs[:, 1:] - sas_log_probs[:, :1] + sa_log_probs[:,:1]

                if writer is not None and self.total_it % 5000 == 0:
                    writer.add_scalar('train/reward penalty', reward_penalty.mean(), global_step=self.total_it)

                src_reward += self.config['penalty_coefficient'] * reward_penalty

            # update rnd predictor on source samples
            if self.config.get('rnd_src_steps_per_iter', 1) > 0:
              self.rnd_src_update(src_replay_buffer,
                            iters=self.config['rnd_src_steps_per_iter'],
                            batch_size=self.config.get('rnd_batch_size', batch_size))

        q_loss_step, q_target = self.update_q_functions(src_state, src_action, src_reward, src_next_state, src_not_done, writer)

        self.q_optimizer.zero_grad()
        q_loss_step.backward()
        self.q_optimizer.step()

        # update policy and temperature
        for p in self.q_funcs.parameters():
            p.requires_grad = False
        pi_loss_step, alpha_loss_step = self.update_policy_and_temp(src_state)
        self.pi_optimizer.zero_grad()
        pi_loss_step.backward()
        self.pi_optimizer.step()
        self.temp_optimizer.zero_grad()
        alpha_loss_step.backward()
        self.temp_optimizer.step()
        for p in self.q_funcs.parameters():
            p.requires_grad = True

        # update classifier
        if self.warm_up_step < self.current_step:
            for _ in range(self.update_rate):
                s, a, s2, r_env, not_done = src_replay_buffer.sample(batch_size)

                # Intrinsic bonus (vs-source) and mixed reward for explorer
                r_int = self._intrinsic_bonus(s, a, s2)
                r = r_env + self.exp_beta * r_int

                # --- Q update ---
                with torch.no_grad():
                    an, logp, _ = self.exp_policy(s2, get_logprob=True)
                    q1t, q2t = self.exp_target_q_funcs(s2, an)
                    q_t = torch.min(q1t, q2t)
                    target = r + not_done * self.discount * (q_t - self._exp_alpha * logp)

                q1, q2 = self.exp_q_funcs(s, a)
                q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
                self.exp_q_optimizer.zero_grad()
                q_loss.backward()
                self.exp_q_optimizer.step()

                # soft update explorer target critics
                self._update_target_net(self.exp_target_q_funcs, self.exp_q_funcs)

                # --- policy + temperature updates ---
                for p in self.exp_q_funcs.parameters(): p.requires_grad = False
                ap, logp, _ = self.exp_policy(s, get_logprob=True)
                q1b, q2b = self.exp_q_funcs(s, ap)
                qv = torch.min(q1b, q2b)
                pi_loss = (self._exp_alpha * logp - qv).mean()
                self.exp_policy_optimizer.zero_grad()
                pi_loss.backward()
                self.exp_policy_optimizer.step()
                for p in self.exp_q_funcs.parameters(): p.requires_grad = True

                if self.config.get('exp_temperature_opt', True):
                    temp_loss = -self._exp_alpha * (logp.detach() + (-self.config['action_dim'])).mean()
                    self.exp_temp_optimizer.zero_grad()
                    temp_loss.backward()
                    self.exp_temp_optimizer.step()

                source_logits, _ = self.classifier.sa_classifier(s)
                target_logits, _ = self.classifier.sas_classifier(torch.cat([s, a, s2], dim=1))

                # Compute the source and target domain losses
                source_loss = F.binary_cross_entropy_with_logits(source_logits, torch.zeros_like(source_logits))
                target_loss = F.binary_cross_entropy_with_logits(target_logits, torch.ones_like(target_logits))
                classifier_loss = source_loss + target_loss

                self.classifier_optimizer.zero_grad()
                classifier_loss.backward()
                self.classifier_optimizer.step()

                source_acc = (torch.sigmoid(source_logits) < 0.5).float().mean()
                target_acc = (torch.sigmoid(target_logits) > 0.5).float().mean()

        if self.current_step % 1000 == 0:
            writer.add_scalar('train/batch_reward', src_reward.mean(), self.current_step)
            writer.add_scalar('train/q_target', q_target.mean(), self.current_step)
            writer.add_scalar('train/critic_loss', q_loss_step.item(), self.current_step)
            writer.add_scalar('train/actor_loss', pi_loss_step.item(), self.current_step)
            writer.add_scalar('train/alpha', self.alpha, self.current_step)
            writer.add_scalar('train/alpha_loss', alpha_loss_step.item(), self.current_step)
            if self.warm_up_step < self.current_step:
                writer.add_scalar('classifier/loss', classifier_loss.item(), self.current_step)
                writer.add_scalar('classifier/source_acc', source_acc.item(), self.current_step)
                writer.add_scalar('classifier/target_acc', target_acc.item(), self.current_step)
                if 'reward_penalty' in locals():
                    writer.add_scalar('classifier/reward_penalty', reward_penalty.mean().item(), self.current_step)

        self.current_step += 1
        self.update_target()
        # ...existing code...

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def save(self, filename):
        torch.save(self.q_funcs.state_dict(), filename + "_critic")
        torch.save(self.q_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.policy.state_dict(), filename + "_actor")
        torch.save(self.policy_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.classifier.state_dict(), filename + "_classifier")
        torch.save(self.classifier_optimizer.state_dict(), filename + "_classifier_optimizer")

    def load(self, filename):
        self.q_funcs.load_state_dict(torch.load(filename + "_critic"))
        self.q_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.policy.load_state_dict(torch.load(filename + "_actor"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.classifier.load_state_dict(torch.load(filename + "_classifier"))
        self.classifier_optimizer.load_state_dict(torch.load(filename + "_classifier_optimizer"))
