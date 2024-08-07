import os

from stable_baselines3.a2c import A2C
from stable_baselines3.ddpg import DDPG
from stable_baselines3.dqn import DQN
from stable_baselines3.ppo import PPO
from stable_baselines3.ppo_lag import PPOLagrangian
from stable_baselines3.ppo_lag import RDPPO_onepolicy, RDPPO_twopolicy
from stable_baselines3.ppo_lag import DPPO
from stable_baselines3.ppo_lag import RPPOLagrangian
from stable_baselines3.sac import SAC
from stable_baselines3.td3 import TD3

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file, "r") as file_handler:
    __version__ = file_handler.read().strip()
