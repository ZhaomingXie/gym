import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

def degree_to_radian(degree):
    return degree / 180.0 * np.pi

class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.qpos_mean = np.zeros(22)
        self.qpos_std = np.ones(22)
        self.qpos_mean[0] = 1.4
        self.qpos_mean[1] = 0
        self.qpos_std[5:22] *= np.pi/2
        self.qpos_std[5] = degree_to_radian(45)
        self.qpos_std[6] = degree_to_radian(75)
        self.qpos_std[7] = degree_to_radian(35)
        self.qpos_std[8] = degree_to_radian(25)
        self.qpos_std[9] = degree_to_radian(60)
        self.qpos_std[10] = degree_to_radian(110)
        self.qpos_std[11] = degree_to_radian(160)
        self.qpos_std[12] = degree_to_radian(25)
        self.qpos_std[13] = degree_to_radian(60)
        self.qpos_std[14] = degree_to_radian(110)
        self.qpos_std[15] = degree_to_radian(160)
        self.qpos_std[16] = degree_to_radian(85)
        self.qpos_std[17] = degree_to_radian(85)
        self.qpos_std[18] = degree_to_radian(90)
        self.qpos_std[19] = degree_to_radian(85)
        self.qpos_std[20] = degree_to_radian(85)
        self.qpos_std[21] = degree_to_radian(90)
        
        self.qvel_std = np.ones(23)
        self.qvel_std[7:12] = 10
        self.qvel_std[12] = 30
        self.qvel_std[13] = 20
        self.qvel_std[14:16] = 10
        self.qvel_std[16] = 30
        self.qvel_std[17] = 20
        self.qvel_std[18:24] = 10
        self.qvel_std[0:3] = np.ones(3) * 10
        self.qvel_std[3:6] = np.ones(3) * 10
        self.prev_contact = np.zeros(2)
        mujoco_env.MujocoEnv.__init__(self, 'humanoid.xml', 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        
        data = self.sim.data
        contact = np.zeros(2)
        for i in range(data.ncon):
            geom1_body = data.contact[i].geom1
            geom2_body = self.sim.model.geom_bodyid[data.contact[i].geom2]
            if geom1_body == 0 and geom2_body == 6:
                contact[0] = 1
            elif geom1_body == 0 and geom2_body == 9:
                contact[1] = 1
        #print(data.cfrc_ext[6][5] - self.prev_contact[0], data.cfrc_ext[9][5] - self.prev_contact[1])

        #print(data.cfrc_ext[6], data.cfrc_ext[9])
        # return np.concatenate([(data.qpos.flat[2:] - self.qpos_mean) / self.qpos_std,
        #                       data.qvel.flat[:] / self.qvel_std,
        #                       np.array([(data.cfrc_ext[6][5] - self.prev_contact[0]) > 0, (data.cfrc_ext[9][5] - self.prev_contact[1]) > 0])*1.0])
        return np.concatenate([(data.qpos.flat[2:] - self.qpos_mean) / self.qpos_std,
                               data.qvel.flat[:] / self.qvel_std,
                               contact])
                               #np.array([(data.cfrc_ext[6][5] > 0), data.cfrc_ext[9][5] > 0])*1.0])
                               #data.cinert.flat[:] / 100,
                               #data.cvel.flat[:] / 100)
                               #data.qfrc_actuator.flat[:] / 100,
                               #data.cfrc_ext.flat[:] / 3000])

    def step(self, a):
        #a[0:3] *= 0
        #a[11:17] *= 0
        #a = a * 0.4
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        alive_bonus = 1.0
        data = self.sim.data
        lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost# + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        #reward = np.exp(-(self.sim.data.qvel[0] - 1)**2)
        #print(data.qfrc_actuator.flat[:])
        #print(quad_ctrl_cost)
        obs = np.copy(self._get_obs())
        #self.prev_contact = np.array([data.cfrc_ext[6][5], data.cfrc_ext[9][5]])
        return obs, reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        #self.init_qvel[0] = 1
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        #print(self.init_qpos)
        self.prev_contact *= 0
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20

class HumanoidCustomEnv(HumanoidEnv):
    def __init__(self):
        self.qpos_mean = np.zeros(22)
        self.qpos_std = np.ones(22)
        self.qpos_mean[0] = 1.4
        self.qpos_mean[1] = 0
        self.qpos_std[5:22] *= np.pi/2
        self.qpos_std[5] = degree_to_radian(45)
        self.qpos_std[6] = degree_to_radian(75)
        self.qpos_std[7] = degree_to_radian(35)
        self.qpos_std[8] = degree_to_radian(25)
        self.qpos_std[9] = degree_to_radian(60)
        self.qpos_std[10] = degree_to_radian(110)
        self.qpos_std[11] = degree_to_radian(160)
        self.qpos_std[12] = degree_to_radian(25)
        self.qpos_std[13] = degree_to_radian(60)
        self.qpos_std[14] = degree_to_radian(110)
        self.qpos_std[15] = degree_to_radian(160)
        self.qpos_std[16] = degree_to_radian(85)
        self.qpos_std[17] = degree_to_radian(85)
        self.qpos_std[18] = degree_to_radian(90)
        self.qpos_std[19] = degree_to_radian(85)
        self.qpos_std[20] = degree_to_radian(85)
        self.qpos_std[21] = degree_to_radian(90)
        
        self.qvel_std = np.ones(23)
        self.qvel_std[7:12] = 10
        self.qvel_std[12] = 30
        self.qvel_std[13] = 20
        self.qvel_std[14:16] = 10
        self.qvel_std[16] = 30
        self.qvel_std[17] = 20
        self.qvel_std[18:24] = 10
        self.qvel_std[0:3] = np.ones(3) * 10
        self.qvel_std[3:6] = np.ones(3) * 10
        self.prev_contact = np.zeros(2)
        mujoco_env.MujocoEnv.__init__(self, 'humanoid_v2.xml', 5)
        utils.EzPickle.__init__(self)

if __name__ == "__main__":
    env = HumanoidCustomEnv()
    import pdb; pdb.set_trace()
    while True:
        env.render()
        env.step(env.action_space.sample()*0)

        import time; time.sleep(0.002)
