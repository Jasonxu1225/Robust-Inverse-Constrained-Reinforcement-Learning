import copy
import os
from abc import ABC
from typing import Any, Callable, Dict, Optional, Type, Union

import random
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from common.cns_visualization import traj_visualization_2d
from stable_baselines3.common.dual_variable import DualVariable
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common import logger
from stable_baselines3.common.vec_env import VecNormalizeWithCost
from scipy import special

class RobustDualValueIteration(ABC):

    def __init__(self,
                 env: Union[GymEnv, str],
                 max_iter: int,
                 n_actions: int,
                 height: int,  # table length
                 width: int,  # table width
                 terminal_states: int,
                 stopping_threshold: float,
                 seed: int,
                 gamma: float = 0.99,
                 v0: float = 0.0,
                 budget: float = 0.,
                 apply_lag: bool = True,
                 penalty_initial_value: float = 1,
                 penalty_learning_rate: float = 0.01,
                 penalty_min_value: Optional[float] = None,
                 penalty_max_value: Optional[float] = None,
                 log_file=None,
                 n_sample = 1,
                 cost_threshold = 0.1,
                 op_strength = 0,
                 op_lr = 0,
                 density_threshold = 0,
                 soft_tau = 0.001,
                 ):
        super(RobustDualValueIteration, self).__init__()
        self.stopping_threshold = stopping_threshold
        self.gamma = gamma
        self.env = env
        self.log_file = log_file
        self.max_iter = max_iter
        self.n_actions = n_actions
        self.terminal_states = terminal_states
        self.v0 = v0
        self.seed = seed
        self.height = height
        self.width = width
        self.penalty_initial_value = penalty_initial_value
        self.penalty_min_value = penalty_min_value
        self.penalty_max_value = penalty_max_value
        self.penalty_learning_rate = penalty_learning_rate
        self.apply_lag = apply_lag
        self.budget = budget
        self.num_timesteps = 0
        self.admissible_actions = None
        self.n_sample = n_sample
        self.cost_threshold = cost_threshold
        self.op_strength = op_strength
        self.op_lr = op_lr
        self.density_threshold = density_threshold
        self.soft_tau = soft_tau
        self._setup_model()

    def _setup_model(self) -> None:
        self.dual = DualVariable(self.budget,
                                 self.penalty_learning_rate,
                                 self.penalty_initial_value,
                                 min_clamp=self.penalty_min_value,
                                 max_clamp=self.penalty_max_value)
        self.v = self.get_init_v()
        self.v_safety = self.get_init_v()
        self.pi = self.get_equiprobable_policy()
        self.pi_safety = self.get_equiprobable_policy()
        self.op_pi = self.get_equiprobable_policy()
        self.q2p = np.zeros((self.height, self.width, self.n_actions, self.n_actions))
        self.q2p_safety = np.zeros((self.height, self.width, self.n_actions, self.n_actions))

    def get_init_v(self):
        v_m = self.v0 * np.ones((self.height, self.width))
        # # Value function of terminal state must be 0
        # v0[self.e_x, self.e_y] = 0
        return v_m

    def get_equiprobable_policy(self):
        pi = 1 / self.n_actions * np.ones((self.height, self.width, self.n_actions))
        return pi

    def learn(self,
              total_timesteps: int,
              cost_function: Union[str, Callable],
              latent_info_str: Union[str, Callable] = '',
              callback=None,):
        print('Begin training safety policy')
        self.policy_evaluation_safety(cost_function)
        self.policy_improvement_safety(cost_function)
        print('Safety policy converge')
        print('Begin training task policy')
        self.policy_evaluation(cost_function)
        self.policy_improvement(cost_function)
        print('Task policy converge')
        for iter in tqdm(range(total_timesteps)):
            self.num_timesteps += 1
            # cumu_reward, length, dual_stable = self.dual_update(cost_function)
            cumu_reward, length = self.test_performance(cost_function)

        # logger.record("policy_stable", policy_stable)
        # logger.record("policy_safety_stable", policy_safety_stable)
        # logger.record("train/iterations", iter)
        logger.record("train/cumulative rewards", cumu_reward)
        logger.record("train/length", length)

    def update_op_strength(self, nominal_obs, cumu_reward, expert_density_matrix,  expert_length, op_lr):
        desity_edge = False
        op_stable = False
        for obs in nominal_obs:
            # how to define the density threshold? And how to compute density?
            if expert_density_matrix[obs[0]][obs[1]] <= self.density_threshold:
                desity_edge = True

        if len(nominal_obs) > expert_length and desity_edge==True and cumu_reward > 0:
            op_stable = True
            if self.op_strength - op_lr >= 0:
                self.op_strength -= op_lr
        else:
            self.op_strength += op_lr
        return op_stable

    def step(self, action):
        return self.env.step(np.asarray([action]))

    def dual_update(self, cost_function):
        """policy rollout for recording training performance"""
        obs = self.env.reset()
        cumu_reward, length = 0, 0
        actions_game, obs_game, costs_game = [], [], []
        while True:
            actions, _ = self.predict(obs=obs, state=None)
            actions_game.append(actions[0])
            obs_primes, rewards, dones, infos = self.step(actions)
            if type(cost_function) is str:
                costs = np.array([info.get(cost_function, 0) for info in infos])
                if isinstance(self.env, VecNormalizeWithCost):
                    orig_costs = self.env.get_original_cost()
                else:
                    orig_costs = costs
            else:
                costs = cost_function(obs, actions)
                orig_costs = costs
            self.admissible_actions = infos[0]['admissible_actions']
            costs_game.append(orig_costs)
            obs = obs_primes
            obs_game.append(obs[0])
            done = dones[0]
            if done:
                break
            cumu_reward += rewards[0]
            length += 1
        costs_game_mean = np.asarray(costs_game).mean()
        self.dual.update_parameter(torch.tensor(costs_game_mean))
        penalty = self.dual.nu().item()
        print("Performance: dual {0}, cost: {1}, states: {2}, "
              "actions: {3}, rewards: {4}.".format(penalty,
                                                   costs_game_mean.tolist(),
                                                   np.asarray(obs_game).tolist(),
                                                   np.asarray(actions_game).tolist(),
                                                   cumu_reward),
              file=self.log_file,
              flush=True)
        dual_stable = True if costs_game_mean == 0 else False
        return cumu_reward, length, dual_stable

    def test_performance(self, cost_function):
        """policy rollout for recording training performance"""
        obs = self.env.reset()
        cumu_reward, length = 0, 0
        actions_game, obs_game, costs_game = [], [], []
        while True:
            actions, _ = self.predict(obs=obs, state=None)
            actions_game.append(actions[0])
            obs_primes, rewards, dones, infos = self.step(actions)
            if type(cost_function) is str:
                costs = np.array([info.get(cost_function, 0) for info in infos])
                if isinstance(self.env, VecNormalizeWithCost):
                    orig_costs = self.env.get_original_cost()
                else:
                    orig_costs = costs
            else:
                costs = cost_function(obs, actions)
                orig_costs = costs
            self.admissible_actions = infos[0]['admissible_actions']
            costs_game.append(orig_costs)
            obs = obs_primes
            obs_game.append(obs[0])
            done = dones[0]
            if done:
                break
            cumu_reward += rewards[0]
            length += 1
        costs_game_mean = np.asarray(costs_game).mean()
        # self.dual.update_parameter(torch.tensor(costs_game_mean))
        # penalty = self.dual.nu().item()
        # print("Performance: dual {0}, cost: {1}, states: {2}, "
        #       "actions: {3}, rewards: {4}.".format(penalty,
        #                                            costs_game_mean.tolist(),
        #                                            np.asarray(obs_game).tolist(),
        #                                            np.asarray(actions_game).tolist(),
        #                                            cumu_reward),
        #       file=self.log_file,
        #       flush=True)
        # dual_stable = True if costs_game_mean == 0 else False
        print("Performance: cost: {0}, states: {1}, "
              "actions: {2}, rewards: {3}.".format(
                                                   costs_game_mean.tolist(),
                                                   np.asarray(obs_game).tolist(),
                                                   np.asarray(actions_game).tolist(),
                                                   cumu_reward),
              file=self.log_file,
              flush=True)
        return cumu_reward, length

    def policy_evaluation(self, cost_function):
        iter = 0

        delta = self.stopping_threshold + 1
        while delta >= self.stopping_threshold and iter <= self.max_iter-1:
            old_v = self.v.copy()
            delta = 0

            # Traverse all states
            for x in range(self.height):
                for y in range(self.width):
                    # Run one iteration of the Bellman update rule for the value function
                    self.optimal_bellman_update(old_v, x, y, cost_function)
                    # Compute difference
                    delta = max(delta, abs(old_v[x, y] - self.v[x, y]))
            iter += 1
        print("\nThe Task Policy Evaluation algorithm converged after {} iterations".format(iter),
              file=self.log_file)

    def policy_evaluation_safety(self, cost_function):
        iter = 0

        delta = self.stopping_threshold + 1
        while delta >= self.stopping_threshold and iter <= self.max_iter-1:
            old_v = self.v_safety.copy()
            delta = 0

            # Traverse all states
            for x in range(self.height):
                for y in range(self.width):
                    # Run one iteration of the Bellman update rule for the value function
                    self.optimal_bellman_update_safety(old_v, x, y, cost_function)
                    # Compute difference
                    delta = max(delta, abs(old_v[x, y] - self.v_safety[x, y]))
            iter += 1
        print("\nThe Safety Policy Evaluation algorithm converged after {} iterations".format(iter),
              file=self.log_file)

    def policy_improvement(self, cost_function):
        q = np.min(self.q2p, axis=3)
        q_safety = np.min(self.q2p_safety, axis=3)
        actions = []
        for action in range(self.n_actions):
            actions.append(action)
        actions = np.array(actions)

        robust_invariant_set = self.get_robust_invariant_set()

        for x in range(self.height):
            for y in range(self.width):
                if [x,y] in robust_invariant_set:
                    for action in range(self.n_actions):
                        if q_safety[x,y,action] < -self.cost_threshold:
                            q[x, y, action] = 0
                    best_action_indices = np.where(q[x, y, actions] ==
                                                   np.max(q[x, y, actions]), True, False)
                    best_actions = actions[np.array(best_action_indices)]
                    self.define_new_policy(x, y, best_actions)
                    # self.pi[x, y, :] = self.softmax(q[x, y, :], self.soft_tau)
                else:
                    policy_prob = copy.copy(self.pi_safety[x, y])
                    if self.admissible_actions is not None:
                        for c_a in range(self.n_actions):
                            if c_a not in self.admissible_actions:
                                policy_prob[c_a] = -float('inf')
                    best_actions = np.argwhere(policy_prob == np.amax(policy_prob)).flatten().tolist()
                    self.define_new_policy(x, y, best_actions)

        return 1

    def policy_improvement_safety(self, cost_function):
        q_safety = np.min(self.q2p_safety, axis=3)
        op_q_safety = np.max(self.q2p_safety, axis=2)
        actions = []
        for action in range(self.n_actions):
            actions.append(action)
        actions = np.array(actions)
        for x in range(self.height):
            for y in range(self.width):
                best_action_indices = np.where(q_safety[x, y, actions] ==
                                               np.max(q_safety[x, y, actions]), True, False)
                best_actions = actions[np.array(best_action_indices)]
                self.define_new_policy_safety(x, y, best_actions)

                worst_action_indices = np.where(op_q_safety[x, y, actions] ==
                                               np.min(op_q_safety[x, y, actions]), True, False)
                worst_actions = actions[np.array(worst_action_indices)]
                self.define_new_op_policy_safety(x, y, worst_actions)
        return 1

    def define_new_policy(self, x, y, best_actions):
        prob = 1 / len(best_actions)

        for a in range(self.n_actions):
            self.pi[x, y, a] = prob if a in best_actions else 0

    def define_new_policy_safety(self, x, y, best_actions):
        prob = 1 / len(best_actions)

        for a in range(self.n_actions):
            self.pi_safety[x, y, a] = prob if a in best_actions else 0

    def define_new_op_policy_safety(self, x, y, worst_actions):
        prob = 1 / len(worst_actions)

        for a in range(self.n_actions):
            self.op_pi[x, y, a] = prob if a in worst_actions else 0

    def optimal_bellman_update(self, old_v, x, y, cost_function):
        if [x, y] in self.terminal_states:
            return
        for action in range(self.n_actions):
            for op_action in range(self.n_actions):
                total_reward = 0
                total_oldv = 0
                for i in range(self.n_sample):
                    states = self.env.reset_with_values(info_dicts=[{'states': [x, y]}])
                    assert states[0][0] == x and states[0][1] == y
                    s_primes, rewards, dones, infos = self.step(action)

                    # opponent
                    op_states = self.env.reset_with_values(info_dicts=[{'states': [x, y]}])
                    assert op_states[0][0] == x and op_states[0][1] == y
                    op_s_primes, op_rewards, op_dones, op_infos = self.step(op_action)

                    total_reward = total_reward + ((1-self.op_strength)*rewards[0] + self.op_strength*op_rewards[0])
                    total_oldv = total_oldv + ((1-self.op_strength)*old_v[s_primes[0][0], s_primes[0][1]]
                                               + self.op_strength*old_v[op_s_primes[0][0], op_s_primes[0][1]])

                avg_reward = total_reward / (1.0*self.n_sample)
                avg_oldv = total_oldv / (1.0*self.n_sample)

                self.q2p[x, y, action, op_action] = (avg_reward + self.gamma * avg_oldv)
        # self.v[x, y] = special.logsumexp(np.min(self.q2p, axis=3), axis=2)[x][y]
        self.v[x, y] = np.max(np.min(self.q2p, axis=3), axis=2)[x][y]

    def optimal_bellman_update_safety(self, old_v, x, y, cost_function):
        if [x, y] in self.terminal_states:
            return
        for action in range(self.n_actions):
            for op_action in range(self.n_actions):
                total_cost = 0
                total_oldv = 0
                for i in range(self.n_sample):
                    states = self.env.reset_with_values(info_dicts=[{'states': [x, y]}])
                    assert states[0][0] == x and states[0][1] == y
                    # Get next state
                    s_primes, rewards, dones, infos = self.step(action)
                    # Get cost from environment.
                    if type(cost_function) is str:
                        costs = np.array([info.get(cost_function, 0) for info in infos])
                        if isinstance(self.env, VecNormalizeWithCost):
                            orig_costs = self.env.get_original_cost()
                        else:
                            orig_costs = costs
                    else:
                        costs = cost_function(states, [action])
                        orig_costs = costs

                    # opponent
                    op_states = self.env.reset_with_values(info_dicts=[{'states': [x, y]}])
                    assert op_states[0][0] == x and op_states[0][1] == y
                    op_s_primes, op_rewards, op_dones, op_infos = self.step(op_action)

                    if type(cost_function) is str:
                        op_costs = np.array([op_info.get(cost_function, 0) for op_info in op_infos])
                        if isinstance(self.env, VecNormalizeWithCost):
                            op_orig_costs = self.env.get_original_cost()
                        else:
                            op_orig_costs = op_costs
                    else:
                        op_costs = cost_function(op_states, [op_action])
                        op_orig_costs = op_costs


                    total_cost = total_cost + ((1-self.op_strength)*orig_costs[0] + self.op_strength*op_orig_costs[0])
                    total_oldv = total_oldv + ((1-self.op_strength)*old_v[s_primes[0][0], s_primes[0][1]]
                                               + self.op_strength*old_v[op_s_primes[0][0], op_s_primes[0][1]])


                avg_cost = total_cost / (1.0*self.n_sample)
                avg_oldv = total_oldv / (1.0*self.n_sample)

                self.q2p_safety[x, y, action, op_action] = (-avg_cost + self.gamma * avg_oldv)
        self.v_safety[x, y] = np.max(np.min(self.q2p_safety, axis=3), axis=2)[x][y]

    def predict(self, obs, state, deterministic=None):
        policy_prob = copy.copy(self.pi[int(obs[0][0]), int(obs[0][1])])
        if self.admissible_actions is not None:
            for c_a in range(self.n_actions):
                if c_a not in self.admissible_actions:
                    policy_prob[c_a] = -float('inf')
        best_actions = np.argwhere(policy_prob == np.amax(policy_prob)).flatten().tolist()
        action = random.choice(best_actions)
        return np.asarray([action]), state
        # action = np.random.choice(len(policy_prob), p=policy_prob)
        # return np.asarray([action]), state

    def mix_predict(self, obs, state, test_op_strength, deterministic=None):
        # opponent take action
        if np.random.random() < test_op_strength:
            policy_prob = copy.copy(self.op_pi[int(obs[0][0]), int(obs[0][1])])
            if self.admissible_actions is not None:
                for c_a in range(self.n_actions):
                    if c_a not in self.admissible_actions:
                        policy_prob[c_a] = -float('inf')
            worst_actions = np.argwhere(policy_prob == np.amax(policy_prob)).flatten().tolist()
            action = random.choice(worst_actions)
        # player take action
        else:
            policy_prob = copy.copy(self.pi[int(obs[0][0]), int(obs[0][1])])
            if self.admissible_actions is not None:
                for c_a in range(self.n_actions):
                    if c_a not in self.admissible_actions:
                        policy_prob[c_a] = -float('inf')
            best_actions = np.argwhere(policy_prob == np.amax(policy_prob)).flatten().tolist()
            action = random.choice(best_actions)

        return np.asarray([action]), state

    def get_robust_invariant_set(self):
        robust_invariant_set = []
        for x in range(self.height):
            for y in range(self.width):
                if np.max(np.min(self.q2p_safety, axis=3), axis=2)[x][y] >= -self.cost_threshold:
                    robust_invariant_set.append([x, y])
        return robust_invariant_set

    def save(self, save_path):
        state_dict = dict(
            pi=self.pi,
            v_m=self.v,
            gamma=self.gamma,
            max_iter=self.max_iter,
            n_actions=self.n_actions,
            terminal_states=self.terminal_states,
            seed=self.seed,
            height=self.height,
            width=self.width,
            budget=self.budget,
            num_timesteps=self.num_timesteps,
            stopping_threshold=self.stopping_threshold,
        )
        torch.save(state_dict, save_path)

    def softmax_probs(self, x):
        return np.exp(x - np.max(x, axis=2).reshape(x.shape[0], x.shape[1], 1)) / np.exp(
            x - np.max(x, axis=2).reshape(x.shape[0], x.shape[1], 1)).sum(axis=2).reshape(x.shape[0], x.shape[1], 1)

    def softmax(self, Q_values, tau):
        """Apply softmax function to an array of Q-values with temperature tau."""
        Q_values_tau = Q_values / tau
        Q_values_exp = np.exp(Q_values_tau - np.max(Q_values_tau))  # for numerical stability
        probabilities = Q_values_exp / np.sum(Q_values_exp)
        return probabilities


def load_pi(model_path, iter_msg, log_file):
    if iter_msg == 'best':
        model_path = os.path.join(model_path, "best_nominal_model")
    else:
        model_path = os.path.join(model_path, 'model_{0}_itrs'.format(iter_msg), 'nominal_agent')
    print('Loading model from {0}'.format(model_path), flush=True, file=log_file)

    state_dict = torch.load(model_path)

    pi = state_dict["pi"]
    v_m = state_dict["v_m"]
    gamma = state_dict["gamma"]
    max_iter = state_dict["max_iter"]
    n_actions = state_dict["n_actions"]
    terminal_states = state_dict["terminal_states"]
    seed = state_dict["seed"]
    height = state_dict["height"]
    width = state_dict["width"]
    budget = state_dict["budget"]
    stopping_threshold = state_dict["stopping_threshold"]

    create_iteration_agent = lambda: RobustDualValueIteration(
        env=None,
        max_iter=max_iter,
        n_actions=n_actions,
        height=height,  # table length
        width=width,  # table width
        terminal_states=terminal_states,
        stopping_threshold=stopping_threshold,
        seed=seed,
        gamma=gamma,
        budget=budget, )
    iteration_agent = create_iteration_agent()
    iteration_agent.pi = pi
    iteration_agent.v_m = v_m

    return iteration_agent


def compute_expert_density_matrix(expert_obs_list, height, width):
    total_length = 0
    sum_m = np.zeros((height, width))
    for expert_obs in expert_obs_list:
        total_length += len(expert_obs)
        for obs in expert_obs:
            sum_m[obs[0]][obs[1]] += 1
    density_m = sum_m / total_length
    return density_m


def compute_mean_expert_length(expert_obs_list):
    total_length = 0
    for expert_obs in expert_obs_list:
        total_length += len(expert_obs)
    return total_length / len(expert_obs_list)
