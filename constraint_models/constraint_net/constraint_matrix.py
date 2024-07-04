import copy
import os
from itertools import accumulate
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.utils import update_learning_rate
from torch import nn
from tqdm import tqdm

from utils.data_utils import idx2vector

class ConstraintMatrix(nn.Module):
    def __init__(
            self,
            obs_dim: int,
            acs_dim: int,
            hidden_sizes: Tuple[int, ...],
            batch_size: int,
            lr_schedule: Callable[[float], float],
            expert_obs: np.ndarray,
            expert_acs: np.ndarray,
            is_discrete: bool,
            task: str = 'ICRL',
            regularizer_coeff: float = 0.,
            obs_select_dim: Optional[Tuple[int, ...]] = None,
            acs_select_dim: Optional[Tuple[int, ...]] = None,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            no_importance_sampling: bool = True,
            per_step_importance_sampling: bool = False,
            clip_obs: Optional[float] = 10.,
            initial_obs_mean: Optional[np.ndarray] = None,
            initial_obs_var: Optional[np.ndarray] = None,
            action_low: Optional[float] = None,
            action_high: Optional[float] = None,
            target_kl_old_new: float = -1,
            target_kl_new_old: float = -1,
            train_gail_lambda: Optional[bool] = False,
            eps: float = 1e-5,
            recon_obs: bool = False,
            env_configs: dict = None,
            device: str = "cpu",
            log_file=None,
    ):
        super(ConstraintMatrix, self).__init__()
        self.task = task
        self.obs_dim = obs_dim
        self.acs_dim = acs_dim
        self.obs_select_dim = obs_select_dim
        self.acs_select_dim = acs_select_dim
        self.expert_obs = expert_obs
        self.expert_acs = expert_acs
        self.hidden_sizes = hidden_sizes
        self.batch_size = batch_size
        self.is_discrete = is_discrete
        self.regularizer_coeff = regularizer_coeff
        self.importance_sampling = not no_importance_sampling
        self.per_step_importance_sampling = per_step_importance_sampling
        self.clip_obs = clip_obs
        self.device = device
        self.eps = eps
        self.recon_obs = recon_obs
        self.env_configs = env_configs
        self.train_gail_lambda = train_gail_lambda

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_class = optimizer_class
        self.lr_schedule = lr_schedule

        self.current_obs_mean = initial_obs_mean
        self.current_obs_var = initial_obs_var
        self.action_low = action_low
        self.action_high = action_high

        self.target_kl_old_new = target_kl_old_new
        self.target_kl_new_old = target_kl_new_old

        self.current_progress_remaining = 1.
        self.log_file = log_file

        self._build()

    def _build(self) -> None:
        self.cost_matrix = np.zeros([self.env_configs['map_height'], self.env_configs['map_width']])
        self.recon_obs = False

    def cost_function(self, obs: np.ndarray, acs: np.ndarray, force_mode: str = None) -> np.ndarray:
        # cost = self.cost_matrix[obs[0][0]][obs[0][1]]
        # return np.array([cost])
        cost = []
        for i in range(obs.shape[0]):
            cost.append(self.cost_matrix[int(obs[i, 0])][int(obs[i, 1])])
        return np.asarray(cost)

    def get(self, nom_size: int, exp_size: int) -> np.ndarray:
        if self.batch_size is None:
            yield np.arange(nom_size), np.arange(exp_size)
        else:
            size = min(nom_size, exp_size)
            expert_indices = np.random.permutation(exp_size)
            nom_indices = np.random.permutation(nom_size)
            start_idx = 0
            while start_idx < size:
                batch_expert_indices = expert_indices[start_idx:start_idx + self.batch_size]
                batch_nom_indices = nom_indices[start_idx:start_idx + self.batch_size]
                yield batch_nom_indices, batch_expert_indices
                start_idx += self.batch_size

    def train_traj_nn(
            self,
            iterations: np.ndarray,
            nominal_obs: np.ndarray,
            nominal_acs: np.ndarray,
            episode_lengths: np.ndarray,
            obs_mean: Optional[np.ndarray] = None,
            obs_var: Optional[np.ndarray] = None,
            env_configs: Dict = None,
            current_progress_remaining: float = 1,
    ) -> Dict[str, Any]:
        # Prepare data
        nominal_obs = np.concatenate(nominal_obs,axis=0)
        expert_obs = np.concatenate(self.expert_obs,axis=0)

        for i in range(len(nominal_obs)):
            is_in = False
            for j in range(len(expert_obs)):
                if np.array_equal(nominal_obs[i], expert_obs[j]):
                    is_in = True
                    break
            if is_in == False:
                self.cost_matrix[nominal_obs[i][0]][nominal_obs[i][1]] = 1
        # regularize
        if self.regularizer_coeff > 0:
            for i in range(self.env_configs['map_height']):
                for j in range(self.env_configs['map_width']):
                    grid = np.array([i,j])
                    for obs in nominal_obs:
                        if np.array_equal(grid, obs):
                            self.cost_matrix[i][j] = np.clip(self.cost_matrix[i][j]-self.regularizer_coeff, 0, 1)
                            break

        bw_metrics = {"backward/test": 'True'}
        return bw_metrics

    def save(self, save_path):
        state_dict = dict(
            matrix=self.cost_matrix,
        )
        th.save(state_dict, save_path)
