import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
from gym.envs.mujoco.loadstep import CassieTrajectory
import random

import sys
sys.path.append('/home/zhaoming/Documents/dev/gym/gym/envs/mujoco')

class CassieEnv(mujoco_env.MujocoEnv, utils.EzPickle):
	def __init__(self):
		self.P = np.array([100 / 25, 100 / 25, 88 / 16, 96 /16, 50 / 50, 100 / 25, 100 /25, 88 /16, 96 /16, 50 /50])
		self.D = np.array([10.0 / 25, 10.0 /25, 8.0 /16, 9.6 /16, 5.0 /50, 10.0 /25, 10.0 /25, 8.0 /16, 9.6 /16, 5.0 /50])
		self.trajectory = CassieTrajectory("/home/zhaoming/Documents/dev/gym/gym/envs/mujoco/trajectory/stepdata.bin")
		self.time = 0
		self.phase = 0
		self.counter = 0
		self.time_limit = 400
		self.max_phase = 28
		mujoco_env.MujocoEnv.__init__(self, 'cassie.xml', 1)
		utils.EzPickle.__init__(self)

	def get_state(self):
		qpos = self.sim.data.qpos
		qvel = self.sim.data.qvel
		ref_pos, ref_vel = self.get_kin_next_state()
		pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
		vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])
		return np.concatenate([qpos[pos_index], qvel[vel_index], ref_pos[pos_index], ref_vel[vel_index]])

	def step_simulation(self, action):
		qpos = self.sim.data.qpos
		qvel = self.sim.data.qvel
		pos_index = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
		vel_index = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]
		ref_pos, ref_vel = self.get_kin_next_state()
		target = action + ref_pos[pos_index]
		control = self.P*(target-qpos[pos_index])-self.D*qvel[vel_index]
		self.do_simulation(control, self.frame_skip)

	def step(self, action):
		for _ in range(60):
			self.step_simulation(action)

		height = self.sim.data.qpos[2]
		self.time += 1
		self.phase += 1
		if self.phase >= self.max_phase:
			self.phase = 0
			self.counter += 1
		done = not(height > 0.4 and height < 3.0) or self.time >= self.time_limit
		reward = self.compute_reward()
		if reward < 0.3:
			done = True
		return self.get_state(), reward, done, {}

	def get_kin_state(self):
		pose = np.copy(self.trajectory.qpos[self.phase*2*30])
		pose[0] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter
		pose[1] = 0
		vel = np.copy(self.trajectory.qvel[self.phase*2*30])
		return pose, vel

	def get_kin_next_state(self):
		phase = (self.phase + 1) % self.max_phase
		pose = np.copy(self.trajectory.qpos[phase*2*30])
		vel = np.copy(self.trajectory.qvel[phase*2*30])
		pose[0] += (self.trajectory.qpos[1681, 0]- self.trajectory.qpos[0, 0])* self.counter
		pose[1] = 0
		return pose, vel

	def compute_reward(self):
		qpos = self.sim.data.qpos
		qvel = self.sim.data.qvel
		ref_pos, ref_vel = self.get_kin_state()
		weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]
		joint_penalty = 0


		joint_index = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
		vel_index = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

		for i in range(10):
			error = weight[i] * (ref_pos[joint_index[i]]-qpos[joint_index[i]])**2
			joint_penalty += error*30

		com_penalty = (ref_pos[0] - qpos[0])**2 + (qpos[1])**2 + (qpos[2]-ref_pos[2])**2

		orientation_penalty = (qpos[4])**2+(qpos[5])**2+(qpos[6])**2

		spring_penalty = (qpos[15])**2+(qpos[29])**2
		spring_penalty *= 1000

		total_reward = 0.5*np.exp(-joint_penalty)+0.3*np.exp(-com_penalty)+0.1*np.exp(-orientation_penalty)+0.1*np.exp(-spring_penalty)

		return total_reward

	def reset(self):
		self.phase = random.randint(0, 27)
		self.time = 0
		self.counter = 0
		qpos, qvel = self.get_kin_state()
		self.set_state(
			qpos, qvel
		)
		return self.get_state()

if __name__ == "__main__":
    env = CassieEnv()
    while True:
        env.render()
        env.step(env.action_space.sample()*0)

        import time; time.sleep(0.002)
