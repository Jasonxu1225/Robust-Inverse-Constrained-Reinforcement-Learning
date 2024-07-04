import json
import os
import sys
import time
import gym
import numpy as np
import datetime
import yaml

cwd = os.getcwd()
sys.path.append(cwd.replace('/interface', ''))
from utils.data_utils import load_ppo_model
from utils.plot_utils import plot_curve
from stable_baselines3.common import logger
from common.cns_evaluation import evaluate_icrl_policy_cost_episode, evaluate_icrl_policy_cost_episode_withop
from generate_data_for_constraint_inference import create_environments

from utils.data_utils import del_and_make, read_args, load_config, process_memory, print_resource
import warnings

warnings.filterwarnings("ignore")


def evaluate(args):
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

    # monitor the memory and running time
    mem_prev, time_prev = print_resource(mem_prev=mem_prev, time_prev=time_prev,
                                         process_name='Loading environment', log_file=log_file)

    # Logger
    if log_file is None:
        ppo_logger = logger.HumanOutputFormat(sys.stdout)
    else:
        ppo_logger = logger.HumanOutputFormat(log_file)

    timesteps = 0.

    # monitor the memory and running time
    mem_prev, time_prev = print_resource(mem_prev=mem_prev, time_prev=time_prev,
                                         process_name='Setting model', log_file=log_file)

    # evaluate
    start_time = time.time()
    print("\nBeginning evaluation", file=log_file, flush=True)
    mean_reward_list = []
    std_reward_list = []
    mean_nc_reward_list= []
    std_nc_reward_list= []
    record_infos_list= []
    mean_cost_list= []
    std_cost_list= []
    violation_rate_list = []

    #--------------------------------------------load model here-----------------------------------------------
    iter_msg = '49'
    # MEICRL
    model_paths = [
        # '../save_model/ICRL-AntWall/train_ICRL_PPO-Lag_AntWall-multi_env-Jan-23-2024-20:26-seed_123',
        # '../save_model/ICRL-AntWall/train_ICRL_PPO-Lag_AntWall-multi_env-Jan-23-2024-20:26-seed_321',
        # '../save_model/ICRL-AntWall/train_ICRL_PPO-Lag_AntWall-multi_env-Jan-23-2024-20:26-seed_666',
        # '../save_model/ICRL-AntWall/train_ICRL_PPO-Lag_AntWall-multi_env-Jan-23-2024-20:26-seed_888',
    ]

    for model_path in model_paths:

        model = load_ppo_model(model_path, iter_msg=iter_msg, log_file=None)

        if iter_msg == 'best':
            env_stats_loading_path = model_path
        else:
            env_stats_loading_path = os.path.join(model_path, 'model_{0}_itrs'.format(iter_msg))
        config_path = config['env']['eval_config_path']
        if config_path is not None:
            with open(config_path, "r") as config_file:
                env_configs = yaml.safe_load(config_file)
        else:
            env_configs = {}
        if 'Noise' in config['env']['eval_env_id']:
            env_configs = {'noise_mean': config['env']['eval_noise_mean'], 'noise_std': config['env']['eval_noise_std'],
                           'noise_seed': 0, 'noise_type': config['env']['noise_type']}

        sampling_env = create_environments(env_id=config['env']['eval_env_id'],
                                  viz_path=None,
                                  test_path=save_test_mother_dir,
                                  model_path=env_stats_loading_path,
                                  group=config['group'],
                                  num_threads=num_threads,
                                  normalize=not config['env']['dont_normalize_obs'],
                                  env_kwargs=env_configs,
                                  testing_env=False,
                                  part_data=debug_mode)
        sampling_env.norm_reward = False

        for itr in range(config['running']['n_iters']):
            # Evaluate:
            # reward on true environment
            save_path = save_model_mother_dir + '/model_{0}_itrs'.format(itr)
            if itr % config['running']['save_every'] == 0:
                del_and_make(save_path)
            else:
                save_path = None

            mean_reward, std_reward, mean_nc_reward, std_nc_reward, record_infos, mean_cost, std_cost, violation_rate, costs = \
                evaluate_icrl_policy_cost_episode(model=model,
                                     env=sampling_env,
                                     render=False,
                                     record_info_names=config['env']["record_info_names"],
                                     n_eval_episodes=config['running']['n_eval_episodes'],
                                     deterministic=False,
                                     cost_info_str=config['env']['cost_info_str'],
                                     save_path=save_path,
                                     test_op_strength=0)

            # # Save
            if itr % config['running']['save_every'] == 0:
                for record_info_name in config['env']["record_info_names"]:
                    plot_record_infos, plot_costs = zip(*sorted(zip(record_infos[record_info_name], costs)))
                    plot_curve(draw_keys=[record_info_name],
                               x_dict={record_info_name: plot_record_infos},
                               y_dict={record_info_name: plot_costs},
                               xlabel=record_info_name,
                               ylabel='cost',
                               save_name=os.path.join(save_path, "{0}".format(record_info_name)),
                               apply_scatter=True
                               )

            # (2) best
            # Update best metrics

            print('mean_reward:', mean_reward)
            print('mean_nc_reward:', mean_nc_reward)
            print('mean_cost:',mean_cost)
            print('std_cost:', std_cost)
            print('violation_rate:',violation_rate)

            mean_reward_list.append(mean_reward)
            std_reward_list.append(std_reward)
            mean_nc_reward_list.append(mean_nc_reward)
            std_nc_reward_list.append(std_nc_reward)
            record_infos_list.append(record_infos)
            mean_cost_list.append(mean_cost)
            std_cost_list.append(std_cost)

            violation_rate_list.append(violation_rate)

    mean_reward, std_reward, mean_nc_reward, std_nc_reward, record_infos, mean_cost, std_cost, mean_violation_rate, std_violation_rate = \
           np.mean(mean_reward_list), \
           np.std(mean_reward_list), \
           np.mean(mean_nc_reward_list), \
           np.std(mean_nc_reward_list), \
           record_infos_list[0], \
           np.mean(mean_cost_list), \
           np.std(mean_cost_list), \
           np.mean(violation_rate_list), \
           np.std(violation_rate_list)
    # Collect metrics
    metrics = {
        "time(m)": (time.time() - start_time) / 60,
        "timesteps": timesteps,
        "true/nc_reward_mean": mean_nc_reward,
        "true/nc_reward_std": std_nc_reward,
        "true/reward_mean": mean_reward,
        "true/reward_std": std_reward,
        "true/episode_cost_mean": mean_cost,
        "true/episode_cost_std": std_cost,
        "true/violation_rate_mean": mean_violation_rate,
        "true/violation_rate_std": std_violation_rate,
    }

    # Log
    if config['verbose'] > 0:
        ppo_logger.write(metrics, {k: None for k in metrics.keys()}, step=itr)

    # monitor the memory and running time
    mem_prev, time_prev = print_resource(mem_prev=mem_prev, time_prev=time_prev,
                                         process_name='Evaluation', log_file=log_file)


if __name__ == "__main__":
    args = read_args()
    evaluate(args)
