import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)
        self.mean_pos = np.array([1, 0, 0, 0, 0])
        self.mean_vel = np.array([0, 0, 0, 0, 0, 0])
        self.std_pos = np.array([1, np.pi, np.pi, np.pi])
        self.std_vel = np.array([1, 1, 1, 1, 1, 1])

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        #print(self.sim.data.qpos.flat[1:])
        self.mean_pos = np.array([1, 0, 0, 0, 0])
        self.mean_vel = np.array([0, 0, 0, 0, 0, 0])
        self.std_pos = np.array([1, 0.2, 2.6, 2.6, 0.78])
        self.std_vel = np.array([2, 2, 4, 10, 10, 10])
        return np.concatenate([
            (self.sim.data.qpos.flat[1:] - self.mean_pos)/self.std_pos,
            (np.clip(self.sim.data.qvel.flat, -10, 10) - self.mean_vel)/self.std_vel, self.has_contact()*np.ones(1)
        ])
        # return np.concatenate([
        #     self.sim.data.qpos.flat[1:],
        #     np.clip(self.sim.data.qvel.flat, -10, 10)
        # ])

    def has_contact(self):
        contact_force = self.sim.data.cfrc_ext.flat[-1]
        #print(self.sim.data.cfrc_ext)
        if abs(contact_force) > 1e-3:
            return True
        else:
            return False

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

