import os

import gym
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv

# ========================================================================== #
# CHEETAH WITH TORQUE CONSTRAINT
# ========================================================================== #

ACTION_TORQUE_THRESHOLD = 0.5
VIOLATIONS_ALLOWED = 100


class HalfCheetahTest(HalfCheetahEnv):
    def reset(self):
        ob = super().reset()
        self.current_timestep = 0
        self.violations = 0
        return ob

    def step(self, action):
        next_ob, reward, done, info = super().step(action)
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
        return next_ob, reward, done, info


# ========================================================================== #
# ========================================================================== #

REWARD_TYPE = 'old'  # Which reward to use, traditional or new one?

ABS_PATH = os.path.abspath(os.path.dirname(__file__))


# =========================================================================== #
#                           Cheetah With Wall Infront                         #
# =========================================================================== #

class HalfCheetahWithObstacle(HalfCheetahEnv):
    """Variant of half-cheetah that includes an obstacle."""

    def __init__(self, xml_file=ABS_PATH + "/xmls/half_cheetah_obstacle.xml"):
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        utils.EzPickle.__init__(self)
        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low,
            high=self.observation_space.high,
            dtype=np.float32
        )

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(
            reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])


# =========================================================================== #
#            Cheetah With Equal Reward of Moving Forwards and Backwards       #
# =========================================================================== #

class HalfCheetahEqual(HalfCheetahEnv):
    """Also returns the `global' position in HalfCheetah."""

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(
            reward_run=reward_run, reward_ctrl=reward_ctrl)


# =========================================================================== #
#                               Cheetah Backward                              #
# =========================================================================== #

class HalfCheetahBackward(HalfCheetahEnv):
    """Also returns the `global' position in HalfCheetah."""

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = -(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(
            reward_run=reward_run, reward_ctrl=reward_ctrl)


# =========================================================================== #
#                   Cheetah With Global Postion Coordinates                   #
# =========================================================================== #

class HalfCheetahWithPos(HalfCheetahEnv):
    """Also returns the `global' position in HalfCheetah."""

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset_with_info(self, info):
        self.sim.reset()
        if 'qpos' in info.keys() and 'qvel' in info.keys():
            qpos = info['qpos']
            qvel = info['qvel']
            self.set_state(qpos, qvel)
            ob = self._get_obs()
        else:
            ob = self.reset_model()
        info['qpos'] = self.sim.data.qpos
        info['qvel'] = self.sim.data.qpos
        return ob, info

    def old_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run

        info = dict(
            reward_run=reward_run,
            reward_ctrl=reward_ctrl,
            xpos=xposafter,
            # qpos=self.sim.data.qpos,
            # qvel=self.sim.data.qvel,
        )

        return reward, info

    def new_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_dist = abs(xposafter)
        reward_run = reward_dist / self.dt

        reward = reward_dist + reward_ctrl
        info = dict(
            reward_run=reward_run,
            reward_ctrl=reward_ctrl,
            reward_dist=reward_dist,
            xpos=xposafter,
            # qpos=self.sim.data.qpos,
            # qvel=self.sim.data.qvel,
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

class HalfCheetahWithTwoPos(HalfCheetahEnv):
    """Also returns the `global' position in HalfCheetah."""

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset_with_info(self, info):
        self.sim.reset()
        if 'qpos' in info.keys() and 'qvel' in info.keys():
            qpos = info['qpos']
            qvel = info['qvel']
            self.set_state(qpos, qvel)
            ob = self._get_obs()
        else:
            ob = self.reset_model()
        info['qpos'] = self.sim.data.qpos
        info['qvel'] = self.sim.data.qpos
        return ob, info

    def old_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run

        info = dict(
            reward_run=reward_run,
            reward_ctrl=reward_ctrl,
            xpos=xposafter,
            # qpos=self.sim.data.qpos,
            # qvel=self.sim.data.qvel,
        )

        return reward, info

    def new_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_dist = abs(xposafter)
        reward_run = reward_dist / self.dt

        reward = reward_dist + reward_ctrl
        info = dict(
            reward_run=reward_run,
            reward_ctrl=reward_ctrl,
            reward_dist=reward_dist,
            xpos=xposafter,
            # qpos=self.sim.data.qpos,
            # qvel=self.sim.data.qvel,
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

class HalfCheetahWithPosNoise(HalfCheetahWithPos):
    """Also returns the `global' position in HalfCheetah."""

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

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset_with_info(self, info):
        self.sim.reset()
        if 'qpos' in info.keys() and 'qvel' in info.keys():
            qpos = info['qpos']
            qvel = info['qvel']
            self.set_state(qpos, qvel)
            ob = self._get_obs()
        else:
            ob = self.reset_model()
        info['qpos'] = self.sim.data.qpos
        info['qvel'] = self.sim.data.qpos
        return ob, info

    def old_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run

        info = dict(
            reward_run=reward_run,
            reward_ctrl=reward_ctrl,
            xpos=xposafter,
            # qpos=self.sim.data.qpos,
            # qvel=self.sim.data.qvel,
        )

        return reward, info

    def new_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_dist = abs(xposafter)
        reward_run = reward_dist / self.dt

        reward = reward_dist + reward_ctrl
        info = dict(
            reward_run=reward_run,
            reward_ctrl=reward_ctrl,
            reward_dist=reward_dist,
            xpos=xposafter,
            # qpos=self.sim.data.qpos,
            # qvel=self.sim.data.qvel,
        )

        return reward, info

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        if self.noise_type == 'external':
            # # noise = - self.rdm.uniform(0,400) * self.noise_std
            # noise = self.rdm.normal(0,1)
            # body_name_torso = 'torso'
            # body_id_torso = self.sim.model.body_name2id(body_name_torso)
            # # noise_torso = self.rdm.normal(self.noise_mean, self.noise_std, 6)
            # force_torso = [noise, 0, noise ,0, 0, 0]
            # # force_torso = noise_torso
            # self.sim.data.xfrc_applied[body_id_torso] = force_torso

            body_name_bthigh = 'bfoot'
            body_id_bthigh = self.sim.model.body_name2id(body_name_bthigh)
            # noise_bthigh = - self.rdm.uniform(0,300) * self.noise_std
            # force_bthigh = [noise, 0, 0, 0, 0, 0]
            noise_bthigh = 10 * self.rdm.normal(self.noise_mean, self.noise_std, 6)
            force_bthigh = noise_bthigh
            self.sim.data.xfrc_applied[body_id_bthigh] = force_bthigh
            #
            body_name_fthigh = 'ffoot'
            body_id_fthigh = self.sim.model.body_name2id(body_name_fthigh)
            # noise_fthigh = - self.rdm.uniform(0,300) * self.noise_std
            # force_fthigh = [noise, 0, 0, 0, 0, 0]
            noise_fthigh = 10 * self.rdm.normal(self.noise_mean, self.noise_std, 6)
            force_fthigh = noise_fthigh
            self.sim.data.xfrc_applied[body_id_fthigh] = force_fthigh

        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        # add noise to the transition function
        if self.noise_type == 'full_random':
            qpos = self.sim.data.qpos + self.rdm.normal(self.noise_mean, self.noise_std, self.model.nq)
            qvel = self.sim.data.qvel + self.rdm.normal(self.noise_mean, self.noise_std, self.model.nv)
            self.set_state(qpos=qpos, qvel=qvel)

        elif self.noise_type == 'partial_random':
            self.sim.data.qpos[0] -= self.rdm.normal(self.noise_mean, self.noise_std)
            qpos = self.sim.data.qpos
            qvel = self.sim.data.qvel
            self.set_state(qpos=qpos, qvel=qvel)

        elif self.noise_type == 'attack':
            # self.sim.data.qpos[0] += self.noise_std * -1 # push backward
            self.sim.data.qpos[0] -= self.rdm.normal(self.noise_mean, self.noise_std)
            qpos = self.sim.data.qpos
            qvel = self.sim.data.qvel
            self.set_state(qpos=qpos, qvel=qvel)
        elif self.noise_type == 'external':
            pass
        else:
            raise ValueError("Unknown Noise Type")

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

class HalfCheetahWithTwoPosNoise(HalfCheetahWithPos):
    """Also returns the `global' position in HalfCheetah."""

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

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset_with_info(self, info):
        self.sim.reset()
        if 'qpos' in info.keys() and 'qvel' in info.keys():
            qpos = info['qpos']
            qvel = info['qvel']
            self.set_state(qpos, qvel)
            ob = self._get_obs()
        else:
            ob = self.reset_model()
        info['qpos'] = self.sim.data.qpos
        info['qvel'] = self.sim.data.qpos
        return ob, info

    def old_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run

        info = dict(
            reward_run=reward_run,
            reward_ctrl=reward_ctrl,
            xpos=xposafter,
            # qpos=self.sim.data.qpos,
            # qvel=self.sim.data.qvel,
        )

        return reward, info

    def new_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_dist = abs(xposafter)
        reward_run = reward_dist / self.dt

        reward = reward_dist + reward_ctrl
        info = dict(
            reward_run=reward_run,
            reward_ctrl=reward_ctrl,
            reward_dist=reward_dist,
            xpos=xposafter,
            # qpos=self.sim.data.qpos,
            # qvel=self.sim.data.qvel,
        )

        return reward, info

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        # xposafter = self.sim.data.qpos[0]
        # should we add noise on the action?
        # qpos = self.sim.data.qpos.flat[:] + self.rdm.normal(self.noise_mean, self.noise_std)
        # qvel = self.sim.data.qvel.flat[:] + self.rdm.normal(self.noise_mean, self.noise_std)

        # add noise to the transition function
        if self.noise_type == 'random':
            qpos = self.sim.data.qpos + self.rdm.normal(self.noise_mean, self.noise_std, self.model.nq)
            qvel = self.sim.data.qvel + self.rdm.normal(self.noise_mean, self.noise_std, self.model.nv)
        elif self.noise_type == 'attack':
            self.sim.data.qpos[0] += self.rdm.random() * self.noise_std * -1 # push backward
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

class HalfCheetahWithPosTest(HalfCheetahWithPos):
    """Environment to test the agent trained in CheetahWithPos using
       constraints."""

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

        # If agent violates constraints, terminate the episode
        if xposafter <= -3:
            print("Violated constraint in the test environment, terminating the episode.", flush=True)
            done = True
            reward = 0

        return ob, reward, done, info
