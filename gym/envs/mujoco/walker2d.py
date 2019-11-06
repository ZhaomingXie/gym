import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        self.left_counter = 0
        self.right_counter = 0
        self.prev_left = True
        self.prev_right = True
        self.contact = 1
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)
        print("initialize")

    def step(self, a):
        #a = np.ones(6)
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        #data = self.sim.data
        #print(data.qfrc_actuator.flat[:])

        #reward = np.exp(-(self.sim.data.qvel[0] - 1)**2)
        print(a)
        return ob, reward, done, {}

    def _get_obs(self):
        mean_pos = np.array([0, 1.25, 0, 0, 0, 0, 0, 0, 0])
        mean_vel = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        std_pos = np.array([1, 1, 1, 2.6, 2.6, 0.78, 2.6, 2.6, 0.78])
        std_vel = np.array([4, 4, 4, 10, 10, 10, 10, 10, 10])
        #print(self.sim.data.qvel)
        qpos = (self.sim.data.qpos - mean_pos) / std_pos
        qvel = (self.sim.data.qvel - mean_vel) / std_vel
        #print(self.get_contact().shape)
        contacts = self.get_contact()
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10), self.contact*np.array([contacts[0]*1.0, contacts[1]*1.0])]).ravel()
        #return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()
        #return np.concatenate([qpos[1:], np.clip(qvel, -10, 10), np.array([self.prev_right*1.0, self.prev_left*1.0])]).ravel()

    def get_contact(self):
        #print(self.sim.data.cfrc_ext.shape)
        right_contact_forces = self.sim.data.cfrc_ext[4]
        left_contact_forces = self.sim.data.cfrc_ext[7]
        #print(self.sim.data.cfrc_ext)
        contacts = np.array([right_contact_forces[5] > 0, left_contact_forces[5] > 0])
        if contacts[0] != self.prev_right:
            self.right_counter += 1
        else:
            self.right_counter = 0
        if contacts[1] != self.prev_left:
            self.left_counter += 1
        else:
            self.left_counter = 0
        if self.left_counter >= 5:
            self.prev_left = contacts[1]
            self.left_counter = 0
        if self.right_counter >= 5:
            self.prev_right = contacts[0]
            self.right_counter = 0
        #print(contacts)
        return contacts

    def reset_model(self):
        #self.init_qvel[0] = 1
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        self.left_counter = 0
        self.right_counter = 0
        contact = self.get_contact()
        self.prev_right = contact[0]
        self.prev_left = contact[1]
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def set_contact(self, contact):
        self.contact = contact

import osqp
import scipy.sparse as sparse
class Walker2dQP(Walker2dEnv):
    def __init__(self):
        super().__init__()

    def step(self, a):
        prob = osqp.OSQP()
        P = np.identity(self.action_space.shape[0])
        P = sparse.csr_matrix(P)
        q = np.zeros(self.action_space.shape[0])
        lb = np.ones(self.action_space.shape[0] + 1) * -1
        lb[-1] = 1
        ub = lb * -1
        ub[-1] = 1
        constraint = np.concatenate([np.identity(self.action_space.shape[0]), np.expand_dims(a, axis=0)])
        constraint = sparse.csr_matrix(constraint)
        prob.setup(P, q, constraint, lb, ub, alpha=1.0, verbose=False)
        res = prob.solve()
        if res.info.status != 'solved':
            ob, reward, done, _, = super().step(np.ones(6))
            done = True
        else:
            ob, reward, done, _, = super().step(res.x)
        return ob, reward, done, {}