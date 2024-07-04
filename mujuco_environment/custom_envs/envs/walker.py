import numpy as np
from gym.envs.mujoco import walker2d
from gym import utils,spaces
from gym.envs.mujoco import mujoco_env
###############################################################################
# TORQUE CONSTRAINTS
###############################################################################

ACTION_TORQUE_THRESHOLD = 0.5
VIOLATIONS_ALLOWED = 100


class Walker2dTest(walker2d.Walker2dEnv):
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


###############################################################################

REWARD_TYPE = 'old'  # Which reward to use, traditional or new one?


# =========================================================================== #
#                    Walker With Global Postion Coordinates                   #
# =========================================================================== #

class WalkerWithPos(walker2d.Walker2dEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def old_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -1e-3 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        alive_bonus = 1
        reward = reward_ctrl + reward_run + alive_bonus

        info = dict(
            reward_run=reward_run,
            reward_ctrl=reward_ctrl,
            x_position=xposafter
        )

        return reward, info

    def new_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -1e-3 * np.square(action).sum()
        # if xposafter < 0:
        #     reward_dist = 1.5 * abs(xposafter)
        # else:
        reward_dist = abs(xposafter)
        reward_run = reward_dist / self.dt

        reward = reward_dist + reward_ctrl
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
        xposafter, height, ang = self.sim.data.qpos[0:3]
        ob = self._get_obs()

        if REWARD_TYPE == 'new':
            reward, info = self.new_reward(xposbefore,
                                           xposafter,
                                           action)
        elif REWARD_TYPE == 'old':
            reward, info = self.old_reward(xposbefore,
                                           xposafter,
                                           action)

        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        return ob, reward, done, info

class WalkerWithPosNoise(WalkerWithPos):
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
        reward_ctrl = -1e-3 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        alive_bonus = 1
        reward = reward_ctrl + reward_run + alive_bonus

        info = dict(
            reward_run=reward_run,
            reward_ctrl=reward_ctrl,
            x_position=xposafter
        )

        return reward, info

    def new_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -1e-3 * np.square(action).sum()
        # if xposafter < 0:
        #     reward_dist = 1.5 * abs(xposafter)
        # else:
        reward_dist = abs(xposafter)
        reward_run = reward_dist / self.dt

        reward = reward_dist + reward_ctrl
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
        # add noise to the transition function
        if self.noise_type == 'random':
            qpos = self.sim.data.qpos + self.rdm.normal(self.noise_mean, self.noise_std, self.model.nq)
            qvel = self.sim.data.qvel + self.rdm.normal(self.noise_mean, self.noise_std, self.model.nv)
        elif self.noise_type == 'attack':
            self.sim.data.qpos[0] += -0.1 # push backward
            qpos = self.sim.data.qpos
            qvel = self.sim.data.qvel
        else:
            raise ValueError("Unknown Noise Type")
        self.set_state(qpos=qpos, qvel=qvel)
        xposafter, height, ang = self.sim.data.qpos[0:3]
        ob = self._get_obs()

        if REWARD_TYPE == 'new':
            reward, info = self.new_reward(xposbefore,
                                           xposafter,
                                           action)
        elif REWARD_TYPE == 'old':
            reward, info = self.old_reward(xposbefore,
                                           xposafter,
                                           action)

        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        return ob, reward, done, info

class WalkerWithAngle(walker2d.Walker2dEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def old_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -1e-3 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        alive_bonus = 1
        reward = reward_ctrl + reward_run + alive_bonus

        info = dict(
            reward_run=reward_run,
            reward_ctrl=reward_ctrl,
            x_position=xposafter
        )

        return reward, info

    def new_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -1e-3 * np.square(action).sum()
        # if xposafter < 0:
        #     reward_dist = 1.5 * abs(xposafter)
        # else:
        reward_dist = abs(xposafter)
        reward_run = reward_dist / self.dt

        reward = reward_dist + reward_ctrl
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
        xposafter, height, ang = self.sim.data.qpos[0:3]
        ob = self._get_obs()

        if REWARD_TYPE == 'new':
            reward, info = self.new_reward(xposbefore,
                                           xposafter,
                                           action)
        elif REWARD_TYPE == 'old':
            reward, info = self.old_reward(xposbefore,
                                           xposafter,
                                           action)
        # info.update({'noise': 0})
        info.update({'thigh': self.sim.data.qpos[3]})
        # info.update({'left_thigh': self.sim.data.qpos[6]})

        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        return ob, reward, done, info

class WalkerWithAngleNoise(WalkerWithAngle):

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
        reward_ctrl = -1e-3 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        alive_bonus = 1
        reward = reward_ctrl + reward_run + alive_bonus

        info = dict(
            reward_run=reward_run,
            reward_ctrl=reward_ctrl,
            x_position=xposafter
        )

        return reward, info

    def new_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -1e-3 * np.square(action).sum()
        # if xposafter < 0:
        #     reward_dist = 1.5 * abs(xposafter)
        # else:
        reward_dist = abs(xposafter)
        reward_run = reward_dist / self.dt

        reward = reward_dist + reward_ctrl
        info = dict(
            reward_run=reward_run,
            reward_ctrl=reward_ctrl,
            reward_dist=reward_dist,
            x_position=xposafter
        )

        return reward, info


    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        # if self.rdm.random() < 0.1:
        #     action = self.rdm.random(action.shape)
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        # add noise to the transition function
        if self.noise_type == 'full_random':
            qpos = self.sim.data.qpos + self.rdm.normal(self.noise_mean, self.noise_std, self.model.nq)
            qvel = self.sim.data.qvel + self.rdm.normal(self.noise_mean, self.noise_std, self.model.nv)
        elif self.noise_type == 'partial_random':
            self.sim.data.qpos[3] += self.rdm.normal(0, 0.01)
            qpos = self.sim.data.qpos
            qvel = self.sim.data.qvel
        elif self.noise_type == 'attack':
            if self.sim.data.qpos[3] < 0:
                # self.sim.data.qpos[3] -= abs(self.rdm.normal(self.noise_mean, self.noise_std))
                self.sim.data.qpos[3] -= abs(self.rdm.uniform(self.noise_mean, self.noise_std))
            else:
                #self.sim.data.qpos[3] += abs(self.rdm.normal(self.noise_mean, self.noise_std))
                self.sim.data.qpos[3] += abs(self.rdm.uniform(self.noise_mean, self.noise_std))
            qpos = self.sim.data.qpos
            qvel = self.sim.data.qvel
        else:
            raise ValueError("Unknown Noise Type")
        self.set_state(qpos=qpos, qvel=qvel)
        height, ang = self.sim.data.qpos[1:3]
        ob = self._get_obs()

        if REWARD_TYPE == 'new':
            reward, info = self.new_reward(xposbefore,
                                           xposafter,
                                           action)
        elif REWARD_TYPE == 'old':
            reward, info = self.old_reward(xposbefore,
                                           xposafter,
                                           action)
        info.update({'thigh': self.sim.data.qpos[3]})

        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        return ob, reward, done, info

class WalkerWithPosTest(WalkerWithPos):
    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter, height, ang = self.sim.data.qpos[0:3]
        ob = self._get_obs()
        reward_ctrl = -1e-3 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        alive_bonus = 1
        if REWARD_TYPE == 'new':
            reward, info = self.new_reward(xposbefore,
                                           xposafter,
                                           action)
        elif REWARD_TYPE == 'old':
            reward, info = self.old_reward(xposbefore,
                                           xposafter,
                                           action)

        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)

        # If agent violates constraint, terminate the episode
        if xposafter <= -3:
            print("Violated constraint in the test environment; terminating episode")
            done = True
            reward = 0

        return ob, reward, done, info

class WalkerWithAngleWithOp(WalkerWithAngle):

    def __init__(self):
        super().__init__()
        ## Adversarial setup
        self._adv_f_bname = ['thigh', 'thigh_left'] # Byte String name of body on which the adversary force will be applied
        bnames = self.model.body_names
        self._adv_bindex = [(bnames.index(i)+1) for i in self._adv_f_bname] # Index of the body on which the adversary force will be applied
        adv_max_force = 5.
        high_adv = np.ones(2*len(self._adv_bindex))*adv_max_force
        low_adv = -high_adv
        self.adv_action_space = spaces.Box(low_adv, high_adv)
        self.pro_action_space = self.action_space

    def _adv_to_xfrc(self, adv_act):
        new_xfrc = self.sim.data.xfrc_applied*0.0
        for i,bindex in enumerate(self._adv_bindex):
            new_xfrc[bindex] = np.array([adv_act[i*2], 0., adv_act[i*2+1], 0., 0., 0.])
        self.sim.data.xfrc_applied[:] = new_xfrc

    def step(self, action):
        if hasattr(action, '__dict__'):
            self._adv_to_xfrc(action.adv)
            a = action.pl
        else:
            a = action
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter, height, ang = self.sim.data.qpos[0:3]
        ob = self._get_obs()

        if REWARD_TYPE == 'new':
            reward, info = self.new_reward(xposbefore,
                                           xposafter,
                                           action)
        elif REWARD_TYPE == 'old':
            reward, info = self.old_reward(xposbefore,
                                           xposafter,
                                           action)

        info.update({'thigh': self.sim.data.qpos[3]})
        info.update({'left_thigh': self.sim.data.qpos[6]})

        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        return ob, reward, done, info