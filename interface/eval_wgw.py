import datetime
import json
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd.replace('/interface', ''))

import os
import sys
import time
import gym
import numpy as np
import datetime
import yaml

cwd = os.getcwd()
sys.path.append(cwd.replace('/interface', ''))
from common.cns_sampler import ConstrainedRLSampler
from common.cns_visualization import traj_visualization_2d, constraint_visualization_2d
from utils.true_constraint_functions import get_true_cost_function

from common.cns_env import make_eval_env

from stable_baselines3.common import logger
from common.cns_evaluation import evaluate_icrl_policy
from stable_baselines3.iteration.policy_interation_lag import load_pi
from stable_baselines3.iteration.robust_dual_policy_interation_traj_constraint import load_robust_pi
from utils.data_utils import del_and_make, read_args, load_config, process_memory, print_resource
import warnings

warnings.filterwarnings("ignore")


def eval(args):
    # load config
    config, debug_mode, log_file_path, partial_data, num_threads, seed = load_config(args)
    if num_threads > 1:
        multi_env = True
        config.update({'multi_env': True})
    else:
        multi_env = False
        config.update({'multi_env': False})

    if log_file_path is not None:
        log_file = open(log_file_path, 'w')
    else:
        log_file = None
    debug_msg = ''
    if debug_mode:
        config['verbose'] = 2  # the verbosity level: 0 no output, 1 info, 2 debug
        if 'PPO' in config.keys():
            config['PPO']['forward_timesteps'] = 200  # 2000
            config['PPO']['n_steps'] = 200
        else:
            config['iteration']['max_iter'] = 2
        config['running']['n_eval_episodes'] = 10
        config['running']['save_every'] = 1
        debug_msg = 'debug-'
    if num_threads is not None:
        config['env']['num_threads'] = num_threads

    # print the current config
    print(json.dumps(config, indent=4), file=log_file, flush=True)

    # init saving dir for the running models
    current_time_date = datetime.datetime.now().strftime('%b-%d-%Y-%H:%M')
    save_model_mother_dir = '{0}/{1}/{5}{2}{3}-{4}-seed_{6}/'.format(
        config['env']['save_dir'],
        config['task'],
        args.config_file.split('/')[-1].split('.')[0],
        '-multi_env' if multi_env else '',
        current_time_date,
        debug_msg,
        seed
    )
    if not os.path.exists('{0}/{1}/'.format(config['env']['save_dir'], config['task'])):
        os.mkdir('{0}/{1}/'.format(config['env']['save_dir'], config['task']))
    if not os.path.exists(save_model_mother_dir):
        os.mkdir(save_model_mother_dir)
    print("Saving to the file: {0}".format(save_model_mother_dir), file=log_file, flush=True)

    # save the running config
    with open(os.path.join(save_model_mother_dir, "model_hyperparameters.yaml"), "w") as hyperparam_file:
        yaml.dump(config, hyperparam_file)

    mem_prev = process_memory()
    time_prev = time.time()

    save_test_mother_dir = os.path.join(save_model_mother_dir, "test/")
    if not os.path.exists(save_test_mother_dir):
        os.mkdir(save_test_mother_dir)

    sampling_env, env_configs = make_eval_env(env_id=config['env']['eval_env_id'],
                                              config_path=config['env']['eval_config_path'],
                                              save_dir=save_test_mother_dir,
                                              group=config['group'],
                                              use_cost=config['env']['use_cost'],
                                              normalize_obs=not config['env']['dont_normalize_obs'],
                                              cost_info_str=config['env']['cost_info_str'],
                                              log_file=log_file,
                                              part_data=partial_data,
                                              # circle_info=config['env']['circle_info'] if 'Circle' in config[
                                              #     'env']['train_env_id'] else None,
                                              max_scene_per_env=config['env']['max_scene_per_env']
                                              if 'max_scene_per_env' in config['env'].keys() else None,
                                              noise_mean=config['env']['eval_noise_mean'] if 'Noise' in config['env'][
                                                  'eval_env_id'] else None,
                                              noise_std=config['env']['eval_noise_std'] if 'Noise' in config['env'][
                                                  'eval_env_id'] else None,
                                              noise_seed=seed if 'Noise' in config['env'][
                                                  'eval_env_id'] else None,
                                              noise_type=config['env']['noise_type'] if 'Noise' in config['env'][
                                                  'eval_env_id'] else None,
                                              )

    # monitor the memory and running time
    mem_prev, time_prev = print_resource(mem_prev=mem_prev, time_prev=time_prev,
                                         process_name='Loading environment', log_file=log_file)

    # Logger
    if log_file is None:
        ppo_logger = logger.HumanOutputFormat(sys.stdout)
    else:
        ppo_logger = logger.HumanOutputFormat(log_file)
    # visualize the cost function for gridworld
    if 'WGW' in config['env']['eval_env_id']:
        ture_cost_function = get_true_cost_function(env_id=config['env']['eval_env_id'],
                                                    env_configs=env_configs)
        constraint_visualization_2d(cost_function=ture_cost_function,
                                    feature_range=config['env']["visualize_info_ranges"],
                                    select_dims=config['env']["record_info_input_dims"],
                                    num_points_per_feature=env_configs['map_height'],
                                    obs_dim=2,
                                    acs_dim=1,
                                    save_path=save_model_mother_dir
                                    )

    # ---------------------------------------load model----------------------------------------
    iter_msg = config['model']['iter_msg']
    model_path = config['model']['model_path']
    if 'RDPI' in model_path:
        model = load_robust_pi(model_path, iter_msg=iter_msg, log_file=None)
    else:
        model = load_pi(model_path, iter_msg=iter_msg, log_file=None)

    # Callbacks
    all_callbacks = []

    timesteps = 0.

    if 'WGW' in config['env']['eval_env_id']:
        sampler = ConstrainedRLSampler(rollouts=10,
                                       store_by_game=True,  # I move the step out
                                       cost_info_str=None,
                                       sample_multi_env=False,
                                       env_id=config['env']['eval_env_id'],
                                       env=sampling_env)

    # monitor the memory and running time
    mem_prev, time_prev = print_resource(mem_prev=mem_prev, time_prev=time_prev,
                                         process_name='Setting model', log_file=log_file)

    # Train
    start_time = time.time()
    print("\nBeginning training", file=log_file, flush=True)
    best_true_reward = -np.inf
    for itr in range(config['running']['n_iters']):

        # monitor the memory and running time
        mem_prev, time_prev = print_resource(mem_prev=mem_prev,
                                             time_prev=time_prev,
                                             process_name='Training PPO model',
                                             log_file=log_file)

        # Evaluate:
        # reward on true environment
        save_path = save_model_mother_dir + '/model_{0}_itrs'.format(itr)
        if itr % config['running']['save_every'] == 0:
            del_and_make(save_path)
        else:
            save_path = None
        mean_reward, std_reward, mean_nc_reward, std_nc_reward, record_infos, costs,_, violation_rate = \
            evaluate_icrl_policy(model=model,
                                 env=sampling_env,
                                 render=False,
                                 record_info_names=config['env']["record_info_names"],
                                 n_eval_episodes=config['running']['n_eval_episodes'],
                                 deterministic=False,
                                 cost_info_str=config['env']['cost_info_str'],
                                 save_path=save_path,
                                 test_op_strength=0)

        # visualize the trajectory for the grid world env
        if 'WGW' in config['env']['eval_env_id'] and itr % config['running']['save_every'] == 0:
            orig_observations, observations, actions, rewards, sum_rewards, lengths = sampler.sample_from_agent(
                policy_agent=model,
                new_env=sampling_env,
                test_op_strength=0
            )
            traj_visualization_2d(config=config,
                                  observations=orig_observations,
                                  save_path=save_path, )


        # Update best metrics
        if mean_nc_reward >= best_true_reward:
            best_true_reward = mean_nc_reward


        # Collect metrics
        metrics = {
            "time(m)": (time.time() - start_time) / 60,
            "run_iter": itr,
            "timesteps": timesteps,
            "true/mean_nc_reward": mean_nc_reward,
            "true/std_nc_reward": std_nc_reward,
            "true/mean_reward": mean_reward,
            "true/std_reward": std_reward,
            "true/per_episode_cost": np.sum(costs)/config['running']['n_eval_episodes'],
            "true/per_step_cost": np.mean(costs),
            "true/violation_rate": violation_rate,
            "best_true/best_reward": best_true_reward,
        }

        # Log
        if config['verbose'] > 0:
            ppo_logger.write(metrics, {k: None for k in metrics.keys()}, step=itr)

        # monitor the memory and running time
        mem_prev, time_prev = print_resource(mem_prev=mem_prev, time_prev=time_prev,
                                             process_name='Evaluation', log_file=log_file)


if __name__ == "__main__":
    args = read_args()
    eval(args)
