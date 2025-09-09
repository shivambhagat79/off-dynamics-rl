import numpy as np
import torch
import gym
import argparse
import os
import random
import math
import time
import copy
import yaml
import json # in case the user want to modify the hyperparameters
import d4rl # used to make offline environments for source domains
import algo.utils as utils

from pathlib                              import Path
from algo.call_algo                       import call_algo
from dataset.call_dataset                 import call_tar_dataset
from envs.mujoco.call_mujoco_env          import call_mujoco_env
from envs.adroit.call_adroit_env          import call_adroit_env
from envs.antmaze.call_antmaze_env        import call_antmaze_env
from envs.infos                           import get_normalized_score
from tensorboardX                         import SummaryWriter


def eval_policy(policy, env, eval_episodes=10, eval_cnt=None):
    eval_env = env

    avg_reward = 0.
    for episode_idx in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            next_state, reward, done, _ = eval_env.step(action)

            avg_reward += reward
            state = next_state
    avg_reward /= eval_episodes

    print("[{}] Evaluation over {} episodes: {}".format(eval_cnt, eval_episodes, avg_reward))

    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./logs")
    parser.add_argument("--policy", default="SAC", help='policy to use')
    parser.add_argument("--env", default="halfcheetah-friction")
    parser.add_argument('--srctype', default="medium", help='dataset type used in the source domain') # only useful when source domain is offline
    parser.add_argument('--tartype', default="medium", help='dataset type used in the target domain') # only useful when target domain is offline
    # support dataset type:
    # source domain: all valid datasets from D4RL
    # target domain: random, medium, medium-expert, expert
    parser.add_argument('--shift_level', default=0.1, help='the scale of the dynamics shift. Note that this value varies on different settins')
    parser.add_argument('--mode', default=0, type=int, help='the training mode, there are four types, 0: online-online, 1: offline-online, 2: online-offline, 3: offline-offline')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--save-model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument('--tar_env_interact_interval', help='interval of interacting with target env', default=10, type=int)
    parser.add_argument('--max_step', default=int(1e6), type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--params', default=None, help='Hyperparameters for the adopted algorithm, ought to be in JSON format')
    parser.add_argument('--train_on_target_only', action='store_true', help='Train SAC exclusively on the target environment for baseline.')

    args = parser.parse_args()

    if args.train_on_target_only:
        args.policy = 'SAC'
        args.mode = -1 # special mode for target only training

    # we support different ways of specifying tasks, e.g., hopper-friction, hopper_friction, hopper_morph_torso_easy, hopper-morph-torso-easy
    if '_' in args.env:
        args.env = args.env.replace('_', '-')

    if 'halfcheetah' in args.env or 'hopper' in args.env or 'walker2d' in args.env or args.env.split('-')[0] == 'ant':
        domain = 'mujoco'
    elif 'pen' in args.env or 'relocate' in args.env or 'door' in args.env or 'hammer' in args.env:
        domain = 'adroit'
    elif 'antmaze' in args.env:
        domain = 'antmaze'
    else:
        raise NotImplementedError
    print(domain)

    call_env = {
        'mujoco': call_mujoco_env,
        'adroit': call_adroit_env,
        'antmaze': call_antmaze_env,
    }

    # determine referenced environment name
    ref_env_name = args.env + '-' + str(args.shift_level)

    if domain == 'antmaze':
        src_env_name = args.env
        src_env_name_config = args.env
    elif domain == 'adroit':
        src_env_name = args.env
        src_env_name_config = args.env.split('-')[0]
    else:
        src_env_name = args.env.split('-')[0]
        src_env_name_config = src_env_name
    tar_env_name = args.env

    # make environments
    if args.mode == 1 or args.mode == 3:
        if domain == 'antmaze':
            src_env_name = src_env_name.split('-')[0]
            src_env_name += '-' + args.srctype + '-v0'
        elif domain == 'adroit':
            src_env_name = src_env_name.split('-')[0]
            src_env_name += '-' + args.srctype + '-v0'
        else:
            src_env_name += '-' + args.srctype + '-v2'
        src_env = None
        src_eval_env = gym.make(src_env_name)
        src_eval_env.seed(args.seed)
    else:
        if 'antmaze' in src_env_name:
            src_env_config = {
                'env_name': src_env_name,
                'shift_level': args.shift_level,
            }
            src_env = call_env[domain](src_env_config)
            src_env.seed(args.seed)
            src_eval_env = call_env[domain](src_env_config)
            src_eval_env.seed(args.seed + 100)
        else:
            src_env_config = {
                'env_name': src_env_name,
                'shift_level': args.shift_level,
            }
            src_env = call_env[domain](src_env_config)
            src_env.seed(args.seed)
            src_eval_env = copy.deepcopy(src_env)
            src_eval_env.seed(args.seed + 100)

    if args.mode == 2 or args.mode == 3:
        tar_env = None
        tar_env_config = {
            'env_name': tar_env_name,
            'shift_level': args.shift_level,
        }
        tar_eval_env = call_env[domain](tar_env_config)
        tar_eval_env.seed(args.seed + 100)
    else:
        tar_env_config = {
            'env_name': tar_env_name,
            'shift_level': args.shift_level,
        }
        tar_env = call_env[domain](tar_env_config)
        tar_env.seed(args.seed)
        tar_eval_env = call_env[domain](tar_env_config)
        tar_eval_env.seed(args.seed + 100)

    if args.mode not in [-1, 0, 1, 2, 3]:
        raise NotImplementedError # cannot support other modes

    policy_config_name = args.policy.lower()

    # load pre-defined hyperparameter config for training
    config_path = f"{str(Path(__file__).parent.absolute())}/config/{domain}/{policy_config_name}/{src_env_name_config}.yaml"
    if not os.path.exists(config_path):
         # Fallback for LIBERTY if specific config not present
        config_path = f"{str(Path(__file__).parent.absolute())}/config/{domain}/{policy_config_name}/default.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)


    if args.params is not None:
        override_params = json.loads(args.params)
        config.update(override_params)
        print('The following parameters are updated to:', args.params)

    print("------------------------------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
    print("------------------------------------------------------------")

    # log path, we use logging with tensorboard
    if args.train_on_target_only:
        outdir = args.dir + '/' + args.policy + '/' + args.env + '-' + str(args.shift_level) + '/baseline_r' + str(args.seed)
    elif args.mode == 1:
        outdir = args.dir + '/' + args.policy + '/' + args.env + '-srcdatatype-' + args.srctype + '-' + str(args.shift_level) + '/r' + str(args.seed)
    elif args.mode == 2:
        outdir = args.dir + '/' + args.policy + '/' + args.env + '-tardatatype-' + args.tartype + '-' + str(args.shift_level) + '/r' + str(args.seed)
    elif args.mode == 3:
        outdir = args.dir + '/' + args.policy + '/' + args.env + '-srcdatatype-' + args.srctype + '-tardatatype-' + args.tartype + '-' + str(args.shift_level) + '/r' + str(args.seed)
    else:
        outdir = args.dir + '/' + args.policy + '/' + args.env + '-' + str(args.shift_level) + '/r' + str(args.seed)
    writer = SummaryWriter('{}/tb'.format(outdir))
    if args.save_model and not os.path.exists("{}/models".format(outdir)):
        os.makedirs("{}/models".format(outdir))

    # seed all
    if src_env is not None: src_env.action_space.seed(args.seed)
    if tar_env is not None: tar_env.action_space.seed(args.seed)
    src_eval_env.action_space.seed(args.seed)
    tar_eval_env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # get necessary information from both domains
    state_dim = src_eval_env.observation_space.shape[0]
    action_dim = src_eval_env.action_space.shape[0]
    max_action = float(src_eval_env.action_space.high[0])
    min_action = -max_action
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # determine shift_level
    if domain == 'mujoco':
        if args.shift_level in ['easy', 'medium', 'hard']:
            shift_level = args.shift_level
        else:
            shift_level = float(args.shift_level)
    else:
        shift_level = args.shift_level

    config.update({
        'env_name': args.env,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'tar_env_interact_interval': int(args.tar_env_interact_interval),
        'max_step': int(args.max_step),
        'shift_level': shift_level,
    })

    policy = call_algo(args.policy, config, args.mode, device)

    # Initialize KMeansStateNovelty if in an online mode with target interaction
    if (args.mode == 0 or args.mode == 1) and not args.train_on_target_only:
        kmeans_state_novelty = utils.KMeansStateNovelty(state_dim)

    ## write logs to record training parameters
    with open(outdir + 'log.txt','w') as f:
        f.write('\n Policy: {}; Env: {}, seed: {}'.format(args.policy, args.env, args.seed))
        for item in config.items():
            f.write('\n {}'.format(item))

    src_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    tar_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)

    # in case that the domain is offline, we directly load its offline data
    if args.mode == 1 or args.mode == 3:
        src_replay_buffer.convert_D4RL(d4rl.qlearning_dataset(src_eval_env))
        if 'antmaze' in args.env:
            src_replay_buffer.reward -= 1.0

    if args.mode == 2 or args.mode == 3:
        tar_dataset = call_tar_dataset(tar_env_name, shift_level, args.tartype)
        tar_replay_buffer.convert_D4RL(tar_dataset)
        if 'antmaze' in args.env:
            tar_replay_buffer.reward -= 1.0

    eval_cnt = 0

    eval_src_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
    eval_tar_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
    eval_cnt += 1

    # The initial state of the current target env episode, required for LIBERTY
    current_episode_initial_state = None

    if args.mode == 0:
        # online-online learning

        src_state, src_done = src_env.reset(), False
        tar_state, tar_done = tar_env.reset(), False
        current_episode_initial_state = tar_state # Initialize s_0

        src_episode_reward, src_episode_timesteps, src_episode_num = 0, 0, 0
        tar_episode_reward, tar_episode_timesteps, tar_episode_num = 0, 0, 0

        # Novelty tracking variables
        total_novel_states = 0
        tar_episode_novel_states = 0
        tar_steps = 0

        for t in range(int(config['max_step'])):
            src_episode_timesteps += 1

            src_action = (
                policy.select_action(np.array(src_state), test=False) + np.random.normal(0, max_action * 0.2, size=action_dim)
            ).clip(-max_action, max_action)

            src_next_state, src_reward, src_done, _ = src_env.step(src_action)
            src_done_bool = float(src_done) if src_episode_timesteps < src_env._max_episode_steps else 0

            if 'antmaze' in args.env:
                src_reward -= 1.0

            src_replay_buffer.add(src_state, src_action, src_next_state, src_reward, src_done_bool)

            src_state = src_next_state
            src_episode_reward += src_reward

            # interaction with tar env
            if t % config['tar_env_interact_interval'] == 0:
                tar_steps += 1
                tar_episode_timesteps += 1
                tar_action = policy.select_action(np.array(tar_state), test=False)

                tar_next_state, tar_reward, tar_done, _ = tar_env.step(tar_action)

                # Check for novelty
                is_novel = kmeans_state_novelty.check_and_update(np.array(tar_next_state))
                if is_novel:
                    total_novel_states += 1
                    tar_episode_novel_states += 1

                tar_done_bool = float(tar_done) if tar_episode_timesteps < tar_env._max_episode_steps else 0

                if 'antmaze' in args.env:
                    tar_reward -= 1.0

                tar_replay_buffer.add(tar_state, tar_action, tar_next_state, tar_reward, tar_done_bool)

                tar_state = tar_next_state
                tar_episode_reward += tar_reward

            policy.train(src_replay_buffer, tar_replay_buffer, current_episode_initial_state, config['batch_size'], writer)

            if (tar_steps > 0 and tar_steps % 250 == 0):
                writer.add_scalar('novelty/total_novel_states', total_novel_states, global_step=tar_steps)

            if src_done:
                print(f"Total T: {t+1} Src Episode Num: {src_episode_num+1} Episode T: {src_episode_timesteps} Reward: {src_episode_reward:.3f}")
                writer.add_scalar('train/source return', src_episode_reward, global_step = t+1)

                src_state, src_done = src_env.reset(), False
                src_episode_reward = 0
                src_episode_timesteps = 0
                src_episode_num += 1

            if tar_done:
                print(f"Total T: {t+1} Tar Episode Num: {tar_episode_num+1} Episode T: {tar_episode_timesteps} Reward: {tar_episode_reward:.3f}")
                writer.add_scalar('train/target return', tar_episode_reward, global_step = t+1)
                train_normalized_score = get_normalized_score(tar_episode_reward, ref_env_name)
                writer.add_scalar('train/target normalized score', train_normalized_score, global_step = t+1)
                writer.add_scalar('novelty/episode_novel_states', tar_episode_novel_states, global_step=tar_episode_num + 1)

                tar_state, tar_done = tar_env.reset(), False
                current_episode_initial_state = tar_state # Update s_0 for the new episode
                tar_episode_reward = 0
                tar_episode_timesteps = 0
                tar_episode_num += 1
                tar_episode_novel_states = 0 # Reset for new episode

            if (t + 1) % config['eval_freq'] == 0:
                src_eval_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
                tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
                writer.add_scalar('test/source return', src_eval_return, global_step = t+1)
                writer.add_scalar('test/target return', tar_eval_return, global_step = t+1)
                eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)
                writer.add_scalar('test/target normalized score', eval_normalized_score, global_step = t+1)

                eval_cnt += 1

                if args.save_model:
                    policy.save('{}/models/model'.format(outdir))
    elif args.train_on_target_only:
        # online learning on target env only for baseline
        tar_state, tar_done = tar_env.reset(), False
        tar_episode_reward, tar_episode_timesteps, tar_episode_num = 0, 0, 0

        for t in range(int(config['max_step'])):
            tar_episode_timesteps += 1

            tar_action = (
                policy.select_action(np.array(tar_state), test=False) + np.random.normal(0, max_action * 0.2, size=action_dim)
            ).clip(-max_action, max_action)

            tar_next_state, tar_reward, tar_done, _ = tar_env.step(tar_action)
            tar_done_bool = float(tar_done) if tar_episode_timesteps < tar_env._max_episode_steps else 0

            if 'antmaze' in args.env:
                tar_reward -= 1.0

            tar_replay_buffer.add(tar_state, tar_action, tar_next_state, tar_reward, tar_done_bool)

            tar_state = tar_next_state
            tar_episode_reward += tar_reward

            policy.train(src_replay_buffer, tar_replay_buffer, None, config['batch_size'], writer)

            if tar_done:
                print(f"Total T: {t+1} Target-only Episode Num: {tar_episode_num+1} Episode T: {tar_episode_timesteps} Reward: {tar_episode_reward:.3f}")
                writer.add_scalar('train/target_only_return', tar_episode_reward, global_step = t+1)
                train_normalized_score = get_normalized_score(tar_episode_reward, ref_env_name)
                writer.add_scalar('train/target_only_normalized_score', train_normalized_score, global_step = t+1)

                tar_state, tar_done = tar_env.reset(), False
                tar_episode_reward = 0
                tar_episode_timesteps = 0
                tar_episode_num += 1

            if (t + 1) % config['eval_freq'] == 0:
                tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
                writer.add_scalar('test/target_only_return', tar_eval_return, global_step = t+1)
                eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)
                writer.add_scalar('test/target_only_normalized_score', eval_normalized_score, global_step = t+1)

                eval_cnt += 1

                if args.save_model:
                    policy.save('{}/models/model'.format(outdir))
    elif args.mode == 1:
        # offline-online learning
        tar_state, tar_done = tar_env.reset(), False
        current_episode_initial_state = tar_state # Initialize s_0
        tar_episode_reward, tar_episode_timesteps, tar_episode_num = 0, 0, 0

        # Novelty tracking variables
        total_novel_states = 0
        tar_episode_novel_states = 0
        tar_steps = 0

        for t in range(int(config['max_step'])):

            # interaction with tar env
            if t % config['tar_env_interact_interval'] == 0:
                tar_steps += 1
                tar_episode_timesteps += 1
                tar_action = policy.select_action(np.array(tar_state), test=False)

                tar_next_state, tar_reward, tar_done, _ = tar_env.step(tar_action)

                # Check for novelty
                is_novel = kmeans_state_novelty.check_and_update(np.array(tar_next_state))
                if is_novel:
                    total_novel_states += 1
                    tar_episode_novel_states += 1

                tar_done_bool = float(tar_done) if tar_episode_timesteps < src_eval_env._max_episode_steps else 0

                if 'antmaze' in args.env:
                    tar_reward -= 1.0

                tar_replay_buffer.add(tar_state, tar_action, tar_next_state, tar_reward, tar_done_bool)

                tar_state = tar_next_state
                tar_episode_reward += tar_reward

            policy.train(src_replay_buffer, tar_replay_buffer, current_episode_initial_state, config['batch_size'], writer)

            if (tar_steps > 0 and tar_steps % 200 == 0):
                writer.add_scalar('novelty/total_novel_states', total_novel_states, global_step=tar_steps)

            if tar_done:
                print(f"Total T: {t+1} Tar Episode Num: {tar_episode_num+1} Episode T: {tar_episode_timesteps} Reward: {tar_episode_reward:.3f}")
                writer.add_scalar('train/target return', tar_episode_reward, global_step = t+1)
                train_normalized_score = get_normalized_score(tar_episode_reward, ref_env_name)
                writer.add_scalar('train/target normalized score', train_normalized_score, global_step = t+1)
                writer.add_scalar('novelty/episode_novel_states', tar_episode_novel_states, global_step=tar_episode_num + 1)

                tar_state, tar_done = tar_env.reset(), False
                current_episode_initial_state = tar_state # Update s_0 for the new episode
                tar_episode_reward = 0
                tar_episode_timesteps = 0
                tar_episode_num += 1
                tar_episode_novel_states = 0 # Reset for new episode

            if (t + 1) % config['eval_freq'] == 0:
                src_eval_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
                tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
                writer.add_scalar('test/source return', src_eval_return, global_step = t+1)
                writer.add_scalar('test/target return', tar_eval_return, global_step = t+1)
                eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)
                writer.add_scalar('test/target normalized score', eval_normalized_score, global_step = t+1)

                eval_cnt += 1

                if args.save_model:
                    policy.save('{}/models/model'.format(outdir))
    elif args.mode == 2:
        # online-offline learning
        # NOTE: LIBERTY and KMeans novelty exploration are not applicable here as there is no online interaction with the target env
        src_state, src_done = src_env.reset(), False
        src_episode_reward, src_episode_timesteps, src_episode_num = 0, 0, 0

        for t in range(int(config['max_step'])):
            src_episode_timesteps += 1

            src_action = (
                policy.select_action(np.array(src_state), test=False) + np.random.normal(0, max_action * 0.2, size=action_dim)
            ).clip(-max_action, max_action)

            src_next_state, src_reward, src_done, _ = src_env.step(src_action)
            src_done_bool = float(src_done) if src_episode_timesteps < src_env._max_episode_steps else 0

            if 'antmaze' in args.env:
                src_reward -= 1.0

            src_replay_buffer.add(src_state, src_action, src_next_state, src_reward, src_done_bool)

            src_state = src_next_state
            src_episode_reward += src_reward

            # The initial state is None since there is no target env interaction
            policy.train(src_replay_buffer, tar_replay_buffer, None, config['batch_size'], writer)

            if src_done:
                print(f"Total T: {t+1} Src Episode Num: {src_episode_num+1} Episode T: {src_episode_timesteps} Reward: {src_episode_reward:.3f}")
                writer.add_scalar('train/source return', src_episode_reward, global_step = t+1)

                src_state, src_done = src_env.reset(), False
                src_episode_reward = 0
                src_episode_timesteps = 0
                src_episode_num += 1

            if (t + 1) % config['eval_freq'] == 0:
                src_eval_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
                tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
                writer.add_scalar('test/source return', src_eval_return, global_step = t+1)
                writer.add_scalar('test/target return', tar_eval_return, global_step = t+1)
                eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)
                writer.add_scalar('test/target normalized score', eval_normalized_score, global_step = t+1)

                eval_cnt += 1

                if args.save_model:
                    policy.save('{}/models/model'.format(outdir))
    else:
        # offline-offline learning
        # NOTE: LIBERTY and KMeans novelty exploration are not applicable here
        for t in range(int(config['max_step'])):
            policy.train(src_replay_buffer, tar_replay_buffer, None, config['batch_size'], writer)

            if (t + 1) % config['eval_freq'] == 0:
                src_eval_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
                tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
                writer.add_scalar('test/source return', src_eval_return, global_step = t+1)
                writer.add_scalar('test/target return', tar_eval_return, global_step = t+1)
                eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)
                writer.add_scalar('test/target normalized score', eval_normalized_score, global_step = t+1)

                eval_cnt += 1

                if args.save_model:
                    policy.save('{}/models/model'.format(outdir))
    writer.close()