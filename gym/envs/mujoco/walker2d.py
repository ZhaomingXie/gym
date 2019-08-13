import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        a = np.ones(6)
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        if reward < 0:
            reward = 0
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        #data = self.sim.data
        #print(data.qfrc_actuator.flat[:])

        #reward = np.exp(-(self.sim.data.qvel[0] - 1)**2)
        return ob, reward, done, {}

    def _get_obs(self):
        mean_pos = np.array([0, 1.25, 0, 0, 0, 0, 0, 0, 0])
        mean_vel = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        std_pos = np.array([1, 1, 1, 2.6, 2.6, 0.78, 2.6, 2.6, 0.78])
        std_vel = np.array([4, 4, 4, 10, 10, 10, 10, 10, 10])
        #print(self.sim.data.qvel)
        qpos = (self.sim.data.qpos - mean_pos) / std_pos
        qvel = (self.sim.data.qvel - mean_vel) / std_vel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        #self.init_qvel[0] = 1
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
