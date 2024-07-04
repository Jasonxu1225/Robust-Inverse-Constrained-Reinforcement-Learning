import numpy as np
from gym.envs.mujoco import swimmer
from gym.envs.mujoco import swimmer_v3

###############################################################################
# TORQUE CONSTRAINTS
###############################################################################

ACTION_TORQUE_THRESHOLD = 0.5
VIOLATIONS_ALLOWED = 100
class SwimmerTest(swimmer.SwimmerEnv):
    def reset(self):
        ob = super().reset()
        self.current_timestep = 0
        self.violations = 0
        return ob

    def step(self, action):
        next_ob, reward, done, infos = super().step(action)
        # This is to handle the edge case where mujoco_env calls
        # step in __init__ without calling reset with a random
        # action
        try:
            self.current_timestep += 1
            if np.any(np.abs(action) > ACTION_TORQUE_THRESHOLD):
                self.violations += 1
            if self.violations > VIOLATIONS_ALLOWED:
                done = True
                reward = 0
        except:
            pass
        return next_ob, reward, done, infos


##############################################################################
REWARD_TYPE = 'old'         # Which reward to use, traditional or new one?

# =========================================================================== #
#                   Swimmer With Global Postion Coordinates                   #
# =========================================================================== #

class SwimmerWithPos(swimmer.SwimmerEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def old_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -1e-4 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run

        info = dict(
                reward_run=reward_run,
                reward_ctrl=reward_ctrl,
                x_position=xposafter
                )

        return reward, info


    def new_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -1e-4 * np.square(action).sum()
        reward_dist = abs(xposafter) - abs(xposbefore)
        reward_run  = reward_dist / self.dt

        if np.sign(xposafter) == np.sign(xposbefore):
            reward = reward_ctrl + reward_run
        else:
            reward = 0

        info = dict(
                reward_run=reward_run,
                reward_ctrl=reward_ctrl,
                reward_dist=reward_dist,
                x_position=xposafter
                )

        return reward, info


    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        if REWARD_TYPE == 'new':
            reward, info = self.new_reward(xposbefore,
                                           xposafter,
                                           action)
        elif REWARD_TYPE == 'old':
            reward, info = self.old_reward(xposbefore,
                                           xposafter,
                                           action)
        done = False

        return ob, reward, done, info

    def get_info(self):
        old_state = self.sim.get_state()
        env_info = dict(
            qpos=self.sim.data.qpos,
            qvel = self.sim.data.qvel,
            ob = self._get_obs(),
        )
        return env_info

    def recover_with_info(self, info):
        qpos = info['qpos']
        qvel = info['qvel']
        self.set_state(qpos, qvel)
        ob = self._get_obs()

        return ob

class SwimmerWithPosNoise(SwimmerWithPos):

    def __init__(self, noise_mean, noise_std, noise_seed, noise_type='random'):
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.noise_type = noise_type
        self.rdm = np.random.RandomState(noise_seed)
        super().__init__()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def old_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -1e-4 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run

        info = dict(
                reward_run=reward_run,
                reward_ctrl=reward_ctrl,
                x_position=xposafter
                )

        return reward, info


    def new_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -1e-4 * np.square(action).sum()
        reward_dist = abs(xposafter) - abs(xposbefore)
        reward_run  = reward_dist / self.dt

        if np.sign(xposafter) == np.sign(xposbefore):
            reward = reward_ctrl + reward_run
        else:
            reward = 0

        info = dict(
                reward_run=reward_run,
                reward_ctrl=reward_ctrl,
                reward_dist=reward_dist,
                x_position=xposafter
                )

        return reward, info


    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        if self.noise_type == 'random':
            qpos = self.sim.data.qpos + self.rdm.normal(self.noise_mean, self.noise_std, self.model.nq)
            qvel = self.sim.data.qvel + self.rdm.normal(self.noise_mean, self.noise_std, self.model.nv)
        elif self.noise_type == 'attack':
            self.sim.data.qpos[0] += self.rdm.random() * self.noise_std * 1 # push forward
            qpos = self.sim.data.qpos
            qvel = self.sim.data.qvel
        else:
            raise ValueError("Unknown Noise Type")
        self.set_state(qpos=qpos, qvel=qvel)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        if REWARD_TYPE == 'new':
            reward, info = self.new_reward(xposbefore,
                                           xposafter,
                                           action)
        elif REWARD_TYPE == 'old':
            reward, info = self.old_reward(xposbefore,
                                           xposafter,
                                           action)
        done = False

        return ob, reward, done, info


#class SwimmerWithPosTest(SwimmerWithPos):
#    def _get_obs(self):
#        return np.concatenate([
#            self.sim.data.qpos.flat,
#            self.sim.data.qvel.flat,
#        ])
#
#    def step(self, action):
#        xposbefore = self.sim.data.qpos[0]
#        self.do_simulation(action, self.frame_skip)
#        xposafter = self.sim.data.qpos[0]
#        ob = self._get_obs()
#        if REWARD_TYPE == 'new':
#            reward, info = self.new_reward(xposbefore,
#                                           xposafter,
#                                           action)
#        elif REWARD_TYPE == 'old':
#            reward, info = self.old_reward(xposbefore,
#                                           xposafter,
#                                           action)
#        done = False
#
#        # If agent violates constraint, terminate the episode
#        if xposafter <= -3:
#            print("Violated constraint in the test environment; terminating episode")
#            done = True
#            reward = 0
#
#        return ob, reward, done, info
#
#class SwimmerWithPos(swimmer_v3.SwimmerEnv):
#    def __init__(self):
#        super(SwimmerWithPos, self).__init__(exclude_current_positions_from_observation=False)
