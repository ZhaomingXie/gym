import os

current_dir = os.path.dirname(os.path.realpath(__file__))
gym_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
os.sys.path.insert(0, gym_root)

import math
import time

from gym.envs.mujoco import mujoco_env
from gym import utils
from gym.envs.mujoco.loadstep import CassieTrajectory
import numpy as np
import torch
import pybullet

DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi


class CassieEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.global_coordinate = True
        self.lower_joint_limit = np.array([-15, -22.5, -50, -167, -140, -22.5, -22.5, -50, -167, -140]) * np.pi / 180
        self.upper_joint_limit = np.array([22.5, 22.5, 80, -37, -30, 15, 22.5, 80, -37, -30]) * np.pi / 180
        self.init_x = 0
        self.actuator_pos_index = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.actuator_vel_index = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]
        self.pos_index = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 20, 21, 22, 23, 28, 29, 30, 34]
        )
        self.vel_index = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 18, 19, 20, 21, 25, 26, 27, 31]
        )
        self.mirror_pos_index = np.array([1, 2,3,4,5,6,21,22,23,28,29,30,34,7,8,9,14,15,16,20])
        self.mirror_vel_index = np.array([0,1,2,3,4,5,19,20,21,25,26,27,31,6,7,8,12,13,14,18])
        self.control_limit = np.array([4.5, 4.5, 12.2, 12.2, 0.9, 4.5, 4.5, 12.2, 12.2, 0.9])
        self.mirror = True
        self.P = np.array(
            [
                100 / 25,
                100 / 25,
                88 / 16,
                96 / 16,
                50 / 50,
                100 / 25,
                100 / 25,
                88 / 16,
                96 / 16,
                50 / 50,
            ]
        )
        self.D = np.array(
            [
                10.0 / 25,
                10.0 / 25,
                8.0 / 16,
                9.6 / 16,
                5.0 / 50,
                10.0 / 25,
                10.0 / 25,
                8.0 / 16,
                9.6 / 16,
                5.0 / 50,
            ]
        )
        self.trajectory = CassieTrajectory(
            os.path.join(current_dir, "trajectory", "stepdata.bin")
        )
        for i in range(841):
            self.trajectory.qpos[i][7:21] = np.copy(self.trajectory.qpos[(i+841)][21:35])
            self.trajectory.qvel[i][6:19] = np.copy(self.trajectory.qpos[(i+841)][19:32])
            self.trajectory.qpos[i][12] = -self.trajectory.qpos[i][12]
            self.trajectory.qpos[i][21:35] = np.copy(self.trajectory.qpos[(i+841)][7:21])
            self.trajectory.qvel[i][19:32] = np.copy(self.trajectory.qpos[(i+841)][6:19])
            self.trajectory.qpos[i][26] = -self.trajectory.qpos[i][26]
        for i in range(1682):
            self.trajectory.qpos[i][2] = 1.05
            #self.trajectory.qvel[i] *= 0
        self.time = 0
        self.phase = 0
        self.counter = 0
        self.time_limit = 400000
        self.max_phase = 28

        # used for rendering
        self.controller_frameskip = 30

        # fake
        self.foot_body_ids = np.zeros(2).astype(np.int)

        mujoco_env.MujocoEnv.__init__(self, "cassie_terrain.xml", 1)
        utils.EzPickle.__init__(self)

        self.foot_body_ids = np.array(
            [
                self.model._body_name2id["left-foot"],
                self.model._body_name2id["right-foot"],
            ]
        )

    def set_mirror(self, mirror):
        self.mirror = mirror

    def get_state(self):
        qpos = np.copy(self.sim.data.qpos)
        qvel = np.copy(self.sim.data.qvel)
        ref_pos, ref_vel = self.get_kin_next_state()

        import pybullet
        w, x, y, z = qpos[3:7]
        roll, pitch, yaw = pybullet.getEulerFromQuaternion((x, y, z, w))
        #roll, pitch, yaw = quaternion2euler(qpos[3:7])
        #print(roll - roll1, pitch - pitch1, yaw - yaw1)
        matrix = np.array([
            [np.cos(-yaw), -np.sin(-yaw)], 
            [np.sin(-yaw), np.cos(-yaw)]
        ])

        #old_qvel = qvel.copy()
        qvel[0:2] = np.dot(matrix, qvel[0:2])
        ##qvel[3:5] = np.dot(matrix, qvel[3:5])
        if not self.global_coordinate:
            yaw = 0
        
        #qpos[3:7] = euler2quat(z=0, y=pitch, x=roll)
        x, y, z, w = pybullet.getQuaternionFromEuler(np.array([roll, pitch, yaw]))
        qpos[3:7] = (w, x, y, z)
        #print("2", pybullet.getEulerFromQuaternion(qpos[3:7]))
        if qpos[3] < 0:
             qpos[3:7] *= -1

        foot_xyzs = self.sim.data.body_xpos[self.foot_body_ids]
        height = self.sim.data.qpos[2] - np.min(foot_xyzs[:, 2])

        state = np.concatenate(
            [qpos[self.pos_index], qvel[self.vel_index], [self.phase/28.0]]
        )

        if self.mirror and self.phase >= 14:
            # ref_vel[1] = -ref_vel[1]
            # ref_euler = quaternion2euler(ref_pos[3:7])
            # ref_euler[0] = -ref_euler[0]
            # ref_euler[2] = -ref_euler[2]
            # ref_pos[3:7] = euler2quat(z=ref_euler[2],y=ref_euler[1],x=ref_euler[0])
            
            # euler = quaternion2euler(qpos[3:7])
            # euler[0] = -euler[0]
            # euler[2] = -euler[2]
            # qpos[3:7] = euler2quat(z=euler[2],y=euler[1],x=euler[0])

            w, x, y, z = qpos[3:7]
            roll, pitch, yaw = pybullet.getEulerFromQuaternion((x, y, z, w))
            roll *= -1
            yaw *= -1
            x, y, z, w = pybullet.getQuaternionFromEuler(np.array([roll, pitch, yaw]))
            qpos[3:7] = (w, x, y, z)
            
            qvel[1] *= -1
            qvel[3] *= -1
            qvel[5] *= -1
            motor_pos = np.zeros(10)
            motor_pos[0:5] = np.copy(qpos[self.actuator_pos_index[5:10]])
            motor_pos[5:10] = np.copy(qpos[self.actuator_pos_index[0:5]])
            motor_pos[0:2] *= -1
            motor_pos[5:7] *= -1
            motor_vel = np.zeros(10)
            motor_vel[0:5] = np.copy(qpos[self.actuator_vel_index[5:10]])
            motor_vel[5:10] = np.copy(qpos[self.actuator_vel_index[0:5]])
            motor_vel[0:2] *= -1
            motor_vel[5:7] *= -1
            qpos[self.actuator_pos_index] = motor_pos
            qvel[self.actuator_vel_index] = motor_vel
            state = np.concatenate([qpos[self.mirror_pos_index], qvel[self.vel_index], [(self.phase % 14)/28.0]])

        if not self.global_coordinate:
            state[0] = 0
        state[1] = height
        return state

    def step_simulation(self, action):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        
        ref_pos, ref_vel = self.get_kin_next_state()

        target = action + ref_pos[self.actuator_pos_index]
        if self.mirror and self.phase >= 14:
            mirror_action = np.zeros(10)
            mirror_action[0:5] = np.copy(action[5:10])
            mirror_action[5:10] = np.copy(action[0:5])
            mirror_action[0] = -mirror_action[0]
            mirror_action[1] = -mirror_action[1]
            mirror_action[5] = -mirror_action[5]
            mirror_action[6] = -mirror_action[6]
            target = mirror_action + ref_pos[self.actuator_pos_index]
        control = self.P * (target - qpos[self.actuator_pos_index]) - self.D * qvel[self.actuator_vel_index]
        # for index in np.where(abs(control) > self.control_limit)[0]:
        #     print("control limit reached", index, control[index])
        self.do_simulation(control, self.frame_skip)
        #self.check_limit()

        if self.viewer is not None:
            self.viewer.cam.lookat[:] = self.sim.data.qpos[0:3]
            super().render("human")

    def step(self, action):
        for _ in range(self.controller_frameskip):
            self.step_simulation(action)

        self.time += 1
        self.phase += 1
        if self.phase >= self.max_phase:
            self.phase = 0
            self.counter += 1

        state = self.get_state()

        height = state[1]
        done = not (height > 0.6 and height < 3.0) or self.time >= self.time_limit
        reward = self.compute_reward()

        # if reward < 0.3:
        #     done = True

        return state, reward, done, {}

    def get_kin_state(self):
        pose = np.copy(self.trajectory.qpos[self.phase * 2 * 30])
        pose[0] += (
            self.trajectory.qpos[1681, 0] - self.trajectory.qpos[0, 0]
        ) * self.counter
        pose[1] = 0
        vel = np.copy(self.trajectory.qvel[self.phase * 2 * 30])
        pose[3] = 1
        pose[4:7] = 0
        pose[7] = 0
        pose[8] = 0
        pose[21] = 0
        pose[22] = 0
        vel[1:6] = 0
        vel[6] = 0
        vel[7] = 0
        vel[19] = 0
        vel[20] = 0
        return pose, vel

    def get_kin_next_state(self):
        phase = (self.phase + 1) % self.max_phase
        pose = np.copy(self.trajectory.qpos[phase * 2 * 30])
        vel = np.copy(self.trajectory.qvel[phase * 2 * 30])
        pose[0] += (
            self.trajectory.qpos[1681, 0] - self.trajectory.qpos[0, 0]
        ) * self.counter
        pose[1] = 0
        pose[3] = 1
        pose[4:7] = 0
        pose[7] = 0
        pose[8] = 0
        pose[21] = 0
        pose[22] = 0
        vel[1:6] = 0
        vel[6] = 0
        vel[7] = 0
        vel[19] = 0
        vel[20] = 0
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
            error = weight[i] * (ref_pos[joint_index[i]] - qpos[joint_index[i]]) ** 2
            joint_penalty += error * 30

        com_penalty = (
            (ref_pos[0] - qpos[0]) ** 2 + (qpos[1]) ** 2 + (qpos[2] - ref_pos[2]) ** 2
        )

        orientation_penalty = (qpos[4]) ** 2 + (qpos[5]) ** 2 + (qpos[6]) ** 2

        spring_penalty = (qpos[15]) ** 2 + (qpos[29]) ** 2
        spring_penalty *= 1000

        total_reward = (
            0.5 * np.exp(-joint_penalty)
            + 0.3 * np.exp(-com_penalty)
            + 0.1 * np.exp(-orientation_penalty)
            + 0.1 * np.exp(-spring_penalty)
        )

        return total_reward

    def reset(self, height_offset=0, phase=None):
        if phase == None:
            self.phase = self.np_random.randint(0, 27)
        else:
            self.phase = phase
        self.time = 0
        self.counter = 0
        qpos, qvel = self.get_kin_state()
        qpos[2] += height_offset
        #qvel[1] = 0.5
        #q = euler2quat(z=-1.57, y=0,x=0)
        #import pybullet
        # x, y, z, w = pybullet.getQuaternionFromEuler((0, 0, np.pi))
        # qpos[3:7] = (w, x, y, z)
        #qvel[3] = 10
        #qpos[0] -= 0.2
        #qvel[1] = -1
        # qvel[0] = -1
        # qvel[1] = 0
        self.set_state(qpos, qvel)
        foot_xyzs = self.sim.data.body_xpos[self.foot_body_ids]
        self.init_x = np.mean(foot_xyzs[:, 0])
        #import time; time.sleep(2)
        return self.get_state()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 0.3
        self.viewer.cam.lookat[2] = 20
        self.viewer.cam.elevation = -20

    def check_limit(self):
        pos_index = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        qpos = self.sim.data.qpos
        violation_index = list(np.where(qpos[pos_index] <= self.lower_joint_limit + 1e-7)[0])
        violation_index +=list(np.where(qpos[pos_index] >= self.upper_joint_limit - 1e-7)[0])
        #print(qpos[pos_index] - self.lower_joint_limit, self.upper_joint_limit-qpos[pos_index])
        for index in (violation_index):
            print(violation_index, qpos[pos_index[index]]-self.lower_joint_limit[index], self.upper_joint_limit[index]-qpos[pos_index[index]])

    # def render(self, mode="human"):
    #     super().render(width=600, height=400, mode=mode)


class CassieStepperEnv(CassieEnv):
    def __init__(self, render=False):
        self.lookahead = 2
        self.next_step_index = 0
        self.target_reached_count = 0
        # self.stop_frames = 0
        self.step_bonus = 0

        self.n_steps = 37
        self.pitch_limit = 25
        self.yaw_limit = 20
        self.tilt_limit = 0
        self.r_range = np.array([0.35, 0.45])

        self.step_radius = 0.10  # xml
        self.step_half_height = 0.1  # xml
        self.rendered_step_count = 10 # xml
        self.initial_height = 20

        self.terrain_info = np.zeros((self.n_steps, 6))
        self.linear_potential = 0
        self.angular_potential = 0
        self.walk_target = np.zeros(3)

        # self.base_phi = DEG2RAD * np.array(
        #     [-10] + [20, -20] * (self.n_steps // 2 - 1) + [10]
        # )
        self.base_phi = DEG2RAD * np.array(
            [-10] + [20, -20] * (self.n_steps // 2 - 1) + [20, -20]
        )
        self.mirror = True

        self.sample_size = 11
        self.yaw_sample_size = 11
        self.pitch_sample_size = 11
        self.r_sample_size = 11
        self.x_tilt_sample_size = 5
        self.y_tilt_sample_size = 5
        self.max_curriculum = 5
        
        self.yaw_samples = np.linspace(0, 0, num=self.yaw_sample_size) * DEG2RAD
        self.pitch_samples = np.linspace(-50, 50, num=self.pitch_sample_size) * DEG2RAD
        #self.r_samples = np.linspace(0.55, 0.8, num=self.r_sample_size)
        self.r_samples = np.linspace(0.35, 0.35, num=self.r_sample_size)
        self.x_tilt_samples = np.linspace(0, 0, num=self.x_tilt_sample_size) * DEG2RAD
        self.y_tilt_samples = np.linspace(0, 0, num=self.y_tilt_sample_size) * DEG2RAD
        
        self.fake_yaw_samples = np.linspace(-20, 20, num=self.yaw_sample_size) * DEG2RAD
        self.fake_pitch_samples = np.linspace(-50, 50, num=self.pitch_sample_size) * DEG2RAD
        self.fake_r_samples = np.linspace(0.35, 0.45, num=self.r_sample_size)
        self.fake_x_tilt_samples = np.linspace(-20, 20, num=self.x_tilt_sample_size) * DEG2RAD
        self.fake_y_tilt_samples = np.linspace(-20, 20, num=self.y_tilt_sample_size) * DEG2RAD
        
        self.pitch_prob = np.ones(self.r_sample_size) / self.r_sample_size
        self.yaw_pitch_prob = np.ones((self.yaw_sample_size, self.pitch_sample_size)) /(self.yaw_sample_size*self.pitch_sample_size)
        self.yaw_pitch_r_prob = np.ones((self.yaw_sample_size, self.pitch_sample_size, self.r_sample_size)) / (self.yaw_sample_size*self.pitch_sample_size * self.r_sample_size)
        self.yaw_pitch_r_tilt_prob = np.ones((self.yaw_sample_size, self.pitch_sample_size, self.r_sample_size, self.x_tilt_sample_size, self.y_tilt_sample_size)) / (self.yaw_sample_size*self.pitch_sample_size * self.r_sample_size * self.x_tilt_sample_size * self.y_tilt_sample_size)
        #self.avg_prob = np.zeros((self.sample_size, self.sample_size))
        self.av_size = 0
        from gym.envs.mujoco.model import ActorCriticNet


        #for markov chain evaluation
        ##for r
        self.r_transition_matrix = np.ones((self.r_sample_size, self.r_sample_size)) / self.r_sample_size
        self.r_histo_matrix = np.zeros((self.r_sample_size, self.r_sample_size))
        self.r_counter = np.ones((self.r_sample_size, self.r_sample_size))
        self.use_markov = False
        self.step_evaluate_reward = 0
        # for continuous markov
        self.r_transition_gaussian_mean = np.ones(11) * self.r_samples[5]
        self.r_transition_gaussian_std = np.ones(11)*100
        #for pitch
        self.pitch_transition_matrix = np.ones((self.pitch_sample_size, self.pitch_sample_size)) / self.pitch_sample_size
        self.pitch_histo_matrix = np.zeros((self.pitch_sample_size, self.pitch_sample_size))
        self.pitch_counter = np.ones((self.pitch_sample_size, self.pitch_sample_size))
        self.step_performance_data = []


        #for two step generation
        self.double_pitch_prob = np.ones((self.pitch_sample_size, self.pitch_sample_size)) / (self.pitch_sample_size**2)
        self.double_generation = False
        from common.controller import CapabilityNet
        self.capacity_net = CapabilityNet(2)
        self.max_performance = 1

        Net = ActorCriticNet
        self.base_model = Net(
            41,  # observation dim
            10,  # action dim
            hidden_layer=[256, 256],
            num_contact=2,
        )
        if self.mirror:
            state_dict = torch.load(os.path.join(current_dir, "Cassie_mirror_v2.pt"))
        else:
            state_dict = torch.load(os.path.join(current_dir, "cassie_gym_seed8.pt"))
        self.base_model.load_state_dict(state_dict)
        self.generate_planks()
        super().__init__()

        self.stop_frames = 3 # 3

        self.all_contact_geom_ids = {
            self.sim.model.geom_bodyid[self.model._geom_name2id[key]]: key
            for key in ["floor", "step1", "step2", "step3", "step4"]
        }

        self.next_next_pitch = 0
        self.next_next_yaw = 0
        self.next_dr = 0.35
        self.next_next_dr = 0.35
        self.temp_states = np.zeros((self.sample_size**2, self.observation_space.shape[0]))

        self.curriculum = 5
        self.global_coordinate = False
        self.test = False


    def set_markov(self, markov):
        self.use_markov = markov

    def update_double_generation(self, double_generation):
        self.double_generation = double_generation

    def generate_step_placements_from_fix_list(
        self,
        pitch_list,
        n_steps=50
    ):

        dr = self.np_random.uniform(0.35, 0.35, size=n_steps)
        dphi = self.np_random.uniform(0, 0, size=n_steps)
        dtheta = pitch_list

        first_x = min(self.sim.data.body_xpos[self.foot_body_ids][:, 0])
        second_x = max(self.sim.data.body_xpos[self.foot_body_ids][:, 0]) - first_x

        dr[0] = first_x
        dphi[0] = 0.0
        dtheta[0] = np.pi / 2

        dr[1] = second_x
        dphi[1] = 0.0
        dtheta[1] = np.pi / 2

        dr[2] = second_x + 0.35
        dphi[2] = 0.0
        dtheta[2] = np.pi / 2

        self.next_pitch = np.pi/2
        self.next_yaw = 0

        x_tilt = self.np_random.uniform(0, 0, size=n_steps)
        y_tilt = self.np_random.uniform(0, 0, size=n_steps)
        x_tilt[0:3] = 0
        y_tilt[0:3] = 0

        dphi = np.cumsum(dphi)
        #print(dr.shape, dtheta.shape, dphi.shape, self.base_phi.shape)
        x_ = dr * np.sin(dtheta) * np.cos(dphi + self.base_phi)
        x_[2:] = np.sign(x_[2:]) * np.minimum(np.maximum(np.abs(x_[2:]), self.step_radius * 3.5), self.r_range[1])
        y_ = dr * np.sin(dtheta) * np.sin(dphi + self.base_phi)
        z_ = dr * np.cos(dtheta)
        x = np.cumsum(x_)
        y = np.cumsum(y_)
        z = np.cumsum(z_) + self.initial_height

        return np.stack((x, y, z, dphi, x_tilt, y_tilt), axis=1)

    def update_difficulty(self, difficulty):
        self.r_samples = np.linspace(difficulty, difficulty, num=self.r_sample_size)

    def change_step_size(self, size):
        for i in range(10):
            self.sim.model.geom_size[i+1, 1]= size

    def sample_from_pitch_transition_matrix(self):
        prob = np.copy(self.pitch_transition_matrix[self.next_pitch_index, :])
        prob /= prob.sum() 
        inds = self.np_random.choice(np.arange(self.pitch_sample_size), p=prob.reshape(-1),size=1,replace=False)
        pitch = self.pitch_samples[inds[0]]
        pitch = np.pi / 2 + pitch
        r = 0.35
        yaw = 0

        self.prev_pitch_index = self.next_pitch_index
        self.prev_pitch = self.next_pitch
        
        self.next_pitch = self.next_next_pitch
        self.next_yaw = self.next_next_yaw
        self.next_dr = np.copy(self.next_next_dr)
        self.next_pitch_index = self.next_next_pitch_index
        
        self.next_next_pitch = pitch
        self.next_next_yaw = yaw
        self.next_next_dr = r
        self.next_next_pitch_index = inds[0]
        self.set_next_next_step_location(self.next_next_pitch, self.next_next_yaw, self.next_next_dr)

    def two_step_generation(self):
        pitch_low = -50 * DEG2RAD
        pitch_high = 50 * DEG2RAD
        yaw_low = -20 * DEG2RAD
        yaw_high = 20 * DEG2RAD
        from torch.optim import Adam
        pitch_list = [0.0, 0.0, 0.0]
        yaw_list = [0.0, 0.0, 0.0]

        pitches = np.zeros(34)
        yaws = np.zeros(34)
        for i in range(17):
            min_loss = 100.0
            for j in range(10):
                temp_pitches = self.np_random.uniform(pitch_low, pitch_high, 2)
                temp_yaws = self.np_random.uniform(yaw_low, yaw_high, 2)
                with torch.no_grad():
                    pitches_torch = torch.tensor(temp_pitches, dtype=torch.float32)
                    
                    ###### loss that include yaw##########
                    #step_torch.tensor(np.concatenate([temp_pitches, temp_yaws]), dtype=torch.float32)
                    
                    loss = (self.capacity_net(pitches_torch)-0.45)**2
                if loss < min_loss:
                    min_loss = loss
                    pitches[i*2:i*2+2] = np.copy(temp_pitches)
                    yaws[i*2:i*2+2] = np.copy(temp_yaws)
        pitches = torch.tensor(pitches, requires_grad=True, dtype=torch.float32)
        yaws = torch.tensor(yaws, requires_grad=True, dtype=torch.float32)
        pitch_optimizer = Adam([pitches], lr=0.01)
        for i in range(10):
            pitch_optimizer.zero_grad()
            loss1 = (self.capacity_net(pitches.reshape(17, 2))-0.45)**2
            #loss2 = (self.capacity_net(pitches[1:-1].reshape(16, 2))-0.45)**2
            
            ###### loss that include yaw##########
            # sequence1 = torch.cat((pitches.reshape(17, 2), yaws.reshape(17,2)),axis=1)
            # sequence2 = torch.cat((pitches[1:-1].reshape(16, 2), yaws[1:-1].reshape(16,2)),axis=1)
            # loss1 = self.capacity_net(sequence1)
            # loss2 = self.capacity_net(sequence2)
            
            loss = loss1.mean()#+loss2.mean()
            loss.backward()
            pitch_optimizer.step()
        pitches = pitches.clamp(pitch_low, pitch_high)
        pitches = pitches.detach().numpy()
            #id0 = (np.abs(pitches[0]-self.pitch_samples)).argmin()
            #id1 = (np.abs(pitches[1]-self.pitch_samples)).argmin()
        for i in range(34):
            #pitch_list.append(self.pitch_samples[id0] + np.pi/2)
            #pitch_list.append(self.pitch_samples[id1] + np.pi/2)
            pitch_list.append(pitches[i] + np.pi/2)
            #pitch_list.append(pitches[i, 1] + np.pi/2)
        self.pitch_list = np.array(pitch_list)
        return self.pitch_list
    
    def sample_next_next_step_fix(self):
        pitch = self.pitch_list[self.next_step_index+1]
        #print(pitch); import time; time.sleep(1)
        r = 0.35
        yaw = 0

        self.prev_pitch = self.next_pitch
        
        self.next_pitch = self.next_next_pitch
        self.next_yaw = self.next_next_yaw
        self.next_dr = np.copy(self.next_next_dr)
        
        self.next_next_pitch = pitch
        self.next_next_yaw = yaw
        self.next_next_dr = r
        self.set_next_next_step_location(self.next_next_pitch, self.next_next_yaw, self.next_next_dr)

    def update_pitch_transition_matrix(self, pitch_transition_matrix):
        self.pitch_transition_matrix = pitch_transition_matrix
        return

    def update_pitch_transition_mean(self, pitch_transition_mean):
        self.pitch_transition_gaussian_mean = pitch_transition_mean

    def update_pitch_transition_std(self, pitch_transition_std):
        self.pitch_transition_gaussian_std = pitch_transition_std

    def reset_pitch_histo_matrix(self):
        self.pitch_histo_matrix *= 0
        self.pitch_counter = np.ones((self.pitch_sample_size, self.pitch_sample_size))

    def reset_step_performance_data(self):
        self.step_performance_data = []
    def get_step_performance_data(self):
        return self.step_performance_data
    def update_double_pitch_prob(self, double_pitch_prob):
        self.double_pitch_prob = double_pitch_prob
    def update_capacity_net(self, capacity_net):
        self.capacity_net.load_state_dict(capacity_net.state_dict())
    def update_max_performance(self, max_performance):
        self.max_performance = max_performance

    def get_pitch_histo_matrix(self):
        #print(self.pitch_counter)
        return np.copy(self.pitch_histo_matrix / self.pitch_counter)

    def generate_planks(self):
        with open("/home/zhaoming/Documents/dev/gym/gym/envs/mujoco/assets/cassie_planks_generated.xml", "w") as f:
            st = ""
            st += """
            <mujoco>
            <worldbody>
            """        
            for i in np.arange(1, self.n_steps):
                st += """
                <body name="step{:d}" pos="10 0 11">
                <geom pos="0 0 0" name="step{:d}" mass="0" size="{:f} {:f} 0.1" type="box" condim='3' conaffinity='15'/>
                </body>
                """.format(i, i, np.random.uniform(0.1, 0.1), 0.5)        
            st += """
                </worldbody>
                </mujoco>
                """
            f.write(st)

    def randomize_terrain(self):
        if self.double_generation:
            print("double_generation")
            pitch_list = self.two_step_generation()
            self.terrain_info = self.generate_step_placements_from_fix_list(pitch_list, n_steps=self.n_steps)
        else:
            self.terrain_info = self.generate_step_placements(
                n_steps=self.n_steps,
                pitch_limit=self.pitch_limit,
                yaw_limit=self.yaw_limit,
                tilt_limit=self.tilt_limit,
            )

        self.walk_target = self.terrain_info[[0], 0:3].mean(axis=0)

        for index in range(self.rendered_step_count):
            pos = self.terrain_info[index, 0:3]
            phi, x_tilt, y_tilt = self.terrain_info[index, 3:6]

            self.model.body_pos[index + 1, :] = -100#pos[:]
            self.model.body_pos[index + 1, 2] -= self.step_half_height
            #self.model.body_quat[index + 1, :] = euler2quat(phi, y_tilt, x_tilt)
            x, y, z, w = pybullet.getQuaternionFromEuler((x_tilt, y_tilt, phi))
            self.model.body_quat[index + 1, :] = (w, x, y, z)

        #self.generate_height_map()

    def sample_height_field(self):
        current_x = self.sim.data.qpos[0]
        height_field_data = np.copy(self.sim.model.hfield_data).reshape(self.sim.model.hfield_ncol[0], self.sim.model.hfield_nrow[0])
        sampled_dx = np.array([-0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        sample_x = current_x + sampled_dx
        height_field = []
        length_per_index = self.sim.model.hfield_size[0][0] * 2 / self.sim.model.hfield_ncol[0]
        for x in sample_x:
            height_field_index = (x - self.sim.model.geom_pos[0][0] + self.sim.model.hfield_size[0][0]) // length_per_index
            if height_field_index < 0 or height_field_index >= self.sim.model.hfield_nrow[0]:
                height_field.append(-3)
            else:
                prev_index = int(height_field_index)
                next_index = int(height_field_index + 1)
                prev_height = height_field_data[0, prev_index] * self.sim.model.hfield_size[0][2] + self.sim.model.geom_pos[0][2] - self.sim.data.qpos[2]
                next_height = height_field_data[0, next_index] * self.sim.model.hfield_size[0][2] + self.sim.model.geom_pos[0][2] - self.sim.data.qpos[2]
                height_field_index = (x - self.sim.model.geom_pos[0][0] + self.sim.model.hfield_size[0][0]) / length_per_index
                height = (height_field_index - prev_index) * next_height + (next_index - height_field_index) * prev_height
                height_field.append(height)
        return np.array(height_field)

    def generate_height_map(self):
        return


    def generate_height_map(self):
        num_steps, _ = self.terrain_info.shape
        height_field_col = self.sim.model.hfield_ncol[0]
        height_field_row = self.sim.model.hfield_nrow[0]
        height_field_size = np.copy(self.sim.model.hfield_size)

        height_field_data = np.copy(self.sim.model.hfield_data)
        height_field_data = height_field_data.reshape(height_field_col, height_field_row)

        z_max = -100
        z_min = 0
        plank_radius = 0.1
        for i in range(num_steps):
            if self.terrain_info[i, 2] > z_max:
                z_max = self.terrain_info[i, 2]
            #if self.terrain_info[i, 2] < z_min:
            #    z_min = self.terrain_info[i, 2]
        self.sim.model.hfield_size[0][2] = z_max - z_min

        self.sim.model.hfield_size[0][0] = (self.terrain_info[-1, 0] - self.terrain_info[0, 0] + plank_radius*2) / 2
        self.sim.model.hfield_size[0][1] = 25.6
        self.sim.model.geom_pos[0][0] = (self.terrain_info[-1, 0] + self.terrain_info[0, 0]) / 2
        self.sim.model.geom_pos[0][2] = z_min

        step_index = 0
        length_per_index = self.sim.model.hfield_size[0][0] * 2 / height_field_col
        while step_index < num_steps - 1:
            starting_height = self.terrain_info[step_index, 2]
            start_index = (self.terrain_info[step_index, 0] - plank_radius - self.sim.model.geom_pos[0][0] + self.sim.model.hfield_size[0][0]) // length_per_index
            if start_index < 0:
                start_index = 0
            end_index = (self.terrain_info[step_index, 0] + plank_radius - self.sim.model.geom_pos[0][0] + self.sim.model.hfield_size[0][0]) // length_per_index
            height_field_data[:, int(start_index):int(end_index)+1] = 0
            height_field_data[256-2:256+2, int(start_index):int(end_index)+1] = (starting_height - z_min) / (z_max - z_min)
            #print(start_index)

            hfield_start_index = (self.terrain_info[step_index, 0] + plank_radius - self.sim.model.geom_pos[0][0] + self.sim.model.hfield_size[0][0]) // length_per_index
            hfield_end_index = (self.terrain_info[step_index+1, 0] - plank_radius - self.sim.model.geom_pos[0][0] + self.sim.model.hfield_size[0][0]) // length_per_index
            for hfield_index in range(int(hfield_start_index)+1, int(hfield_end_index)):
                fraction = (hfield_index - hfield_start_index-1)/(hfield_end_index-hfield_start_index-1)
                if self.terrain_info[step_index, 2] < self.terrain_info[step_index+1, 2]:
                    height = self.terrain_info[step_index, 2] * (1-fraction)+self.terrain_info[step_index+1, 2]*fraction
                else:
                    height = self.terrain_info[step_index, 2] * (1-fraction)+self.terrain_info[step_index+1, 2]*(fraction)
                    #height = self.terrain_info[step_index+1, 2]
                height = (height - z_min) / (z_max - z_min)
                height_field_data[:, hfield_index] = 0
                height_field_data[256-2:256+2, hfield_index] = height
            step_index += 1

        end_height = self.terrain_info[-1, 2]
        start_index = (self.terrain_info[-1, 0] - plank_radius - self.sim.model.geom_pos[0][0] + self.sim.model.hfield_size[0][0]) // length_per_index
        end_index = (self.terrain_info[-1, 0] + plank_radius - self.sim.model.geom_pos[0][0] + self.sim.model.hfield_size[0][0]) // length_per_index
        height_field_data[256-10:256+10, int(start_index):int(end_index)+1] = (end_height - z_min) / (z_max - z_min)
        self.sim.model.hfield_data[:] = height_field_data.reshape(height_field_col*height_field_row)
        self.sim.model.geom_pos[0][2] = z_min

    def generate_step_placements(
        self,
        n_steps=50,
        yaw_limit=30,
        pitch_limit=25,
        tilt_limit=10,
    ):

        yaw_test = 0
        p_test = 50
        r_test = 0.9
        y_range = np.array([-yaw_test, -yaw_test]) * DEG2RAD
        p_range = np.array([90 - self.curriculum*10, 90 + self.curriculum*10]) * DEG2RAD
        t_range = np.array([-tilt_limit, tilt_limit]) * DEG2RAD

        dr = self.np_random.uniform(0.35, 0.35, size=n_steps)
        dphi = self.np_random.uniform(*y_range, size=n_steps)
        dtheta = self.np_random.uniform(*p_range, size=n_steps)
        #print(dtheta)

        first_x = min(self.sim.data.body_xpos[self.foot_body_ids][:, 0])
        second_x = max(self.sim.data.body_xpos[self.foot_body_ids][:, 0]) - first_x

        dr[0] = first_x
        dphi[0] = 0.0
        dtheta[0] = np.pi / 2

        dr[1] = second_x
        dphi[1] = 0.0
        dtheta[1] = np.pi / 2

        dr[2] = second_x + 0.35
        dphi[2] = 0.0
        dtheta[2] = np.pi / 2

        # test_r = 0.6
        # dr[3] = test_r
        # dphi[3] = 0.0
        # dtheta[3] = (90 - 50) * DEG2RAD

        self.next_pitch = np.pi/2
        self.next_yaw = 0
        self.pitch_list = np.copy(dtheta)

        x_tilt = self.np_random.uniform(*t_range, size=n_steps)
        y_tilt = self.np_random.uniform(*t_range, size=n_steps)
        x_tilt[0:3] = 0
        y_tilt[0:3] = 0

        dphi = np.cumsum(dphi)

        # x_ = dr * np.sin(dtheta) * np.cos(dphi + self.base_phi)
        # y_ = dr * np.sin(dtheta) * np.sin(dphi + self.base_phi)
        # z_ = dr * np.cos(dtheta)

        # # Prevent steps from overlapping
        # np.clip(x_[2:], a_min=self.step_radius * 3.5, a_max=self.r_range[1], out=x_[2:])

        # x = np.cumsum(x_)
        # y = np.cumsum(y_)
        # z = np.cumsum(z_) + self.initial_height

        # min_z = self.step_radius * np.sin(self.tilt_limit * DEG2RAD) + 0.01
        # np.clip(z, a_min=min_z, a_max=None, out=z)
        x_ = dr * np.sin(dtheta) * np.cos(dphi + self.base_phi)
        x_[2:] = np.sign(x_[2:]) * np.minimum(np.maximum(np.abs(x_[2:]), self.step_radius * 3.5), self.r_range[1])
        y_ = dr * np.sin(dtheta) * np.sin(dphi + self.base_phi)
        z_ = dr * np.cos(dtheta)
        x = np.cumsum(x_)
        y = np.cumsum(y_)
        z = np.cumsum(z_) + self.initial_height

        return np.stack((x, y, z, dphi, x_tilt, y_tilt), axis=1)

    def update_steps(self):
        threshold = int(self.rendered_step_count // 2)
        if self.next_step_index >= threshold:
            oldest = (self.next_step_index - threshold - 1) % self.rendered_step_count

            next = min(
                (self.next_step_index - threshold - 1) + self.rendered_step_count,
                len(self.terrain_info) - 1,
            )
            pos = self.terrain_info[next, 0:3]
            phi, x_tilt, y_tilt = self.terrain_info[next, 3:6]

            # +1 because first body is worldBody
            self.model.body_pos[oldest + 1, :] = -100#pos[:]
            # account for half height
            self.model.body_pos[oldest + 1, 2] -= self.step_half_height
            #self.model.body_quat[oldest + 1, :] = euler2quat(phi, y_tilt, x_tilt)
            x, y, z, w = pybullet.getQuaternionFromEuler((x_tilt, y_tilt, phi))
            self.model.body_quat[oldest + 1, :] = (w, x, y, z)

    def reset(self):

        self.next_step_index = 1
        self.target_reached_count = 0

        self.next_next_pitch = np.pi/2
        self.next_next_yaw = 0
        self.next_dr = 0.35
        self.next_next_dr = 0.35
        self.next_x_tilt = 0
        self.next_next_x_tilt = 0
        self.next_y_tilt = 0
        self.next_next_y_tilt = 0

        self.next_r_index = 0
        self.next_next_r_index = 0
        self.prev_r_index = 0
        self.next_pitch_index = 5
        self.next_next_pitch_index = 5
        self.prev_pitch_index = 5

        self.prev_pitch = np.pi/2

        obs = super().reset(height_offset=self.initial_height, phase=9)
        self.randomize_terrain()
        self.generate_height_map()

        self.targets = self.delta_to_k_targets(k=self.lookahead)

        self.calc_potential()
        state = np.concatenate((obs, self.targets.flatten()))

        self.step_evaluate_reward = 0

        self.av_size = 0

        return state

    def step(self, action):
        obs = torch.from_numpy(super().get_state()).float()

        with torch.no_grad():
            base_action = self.base_model.sample_best_actions(obs).squeeze().numpy()

        obs, _, done, _ = super().step(action + base_action)

        # check if target changed
        cur_step_index = self.next_step_index

        self.calc_progress_reward()
        self.calc_step_state()

        if self.use_markov or self.double_generation:
            self.step_evaluate_reward += self.step_bonus

        self.update_terrain = (cur_step_index != self.next_step_index)

        if self.use_markov and self.update_terrain:
            self.pitch_histo_matrix[self.prev_pitch_index, self.next_pitch_index] += self.step_evaluate_reward
            self.pitch_counter[self.prev_pitch_index, self.next_pitch_index] += 1
            # self.step_performance_data.append(np.array([self.prev_pitch, self.next_pitch, self.step_evaluate_reward]))
            self.step_evaluate_reward = 0

        if self.double_generation and self.update_terrain:
            if self.step_evaluate_reward / 60 > -0.1:
                self.step_performance_data.append(np.array([self.prev_pitch-np.pi/2, self.next_pitch-np.pi/2, self.step_evaluate_reward]))
                self.step_performance_data.append(np.array([self.next_pitch-np.pi/2, self.next_next_pitch-np.pi/2, self.step_evaluate_reward]))
            self.step_evaluate_reward = 0
        elif self.double_generation and done == True:
            #print(self.step_evaluate_reward)
            if self.step_evaluate_reward / 60 > -0.1:
                self.step_performance_data.append(np.array([self.prev_pitch-np.pi/2, self.next_pitch-np.pi/2, self.step_evaluate_reward]))
                self.step_performance_data.append(np.array([self.next_pitch-np.pi/2, self.next_next_pitch-np.pi/2, self.step_evaluate_reward]))
            self.step_evaluate_reward = 0

        if cur_step_index != self.next_step_index:
            # needs to be done after walk_target is updated
            # which is in delta_to_k_targets()
            self.update_terrain_info()
            self.calc_potential()

        self.targets = self.delta_to_k_targets(k=self.lookahead)
        obs_target = np.copy(self.targets)
        if self.mirror and self.phase >= 14:
            obs_target[:, [0, 3]] *= -1
        state = np.concatenate((obs, obs_target.flatten()))

        orientation_penalty = np.sum(self.sim.data.qvel[3:6]**2)


        return (state, self.progress + self.step_bonus - orientation_penalty * 0.0, done, {})

    def update_terrain_info(self):
        # print(env.next_step_index)
        next_next_step = self.next_step_index + 1 

        if self.double_generation:
            self.sample_next_next_step_fix()
        elif not self.use_markov:
            self.sample_next_next_step_fix()
        else:
            self.sample_from_pitch_transition_matrix()
            #self.sample_from_r_transition_matrix()
        # +1 because first body is worldBody
        body_index = next_next_step % self.rendered_step_count + 1
        self.model.body_pos[body_index, :] = -100#self.terrain_info[next_next_step, 0:3]
        # account for half height
        self.model.body_pos[body_index, 2] -= self.step_half_height    
        
        phi, x_tilt, y_tilt = self.terrain_info[next_next_step, 3:6]
        x, y, z, w = pybullet.getQuaternionFromEuler((x_tilt, y_tilt, phi))
        self.model.body_quat[body_index, :] = (w, x, y, z)
        self.targets = self.delta_to_k_targets(k=self.lookahead)


    def get_temp_state(self):
        obs = self.get_state()
        target = self.delta_to_k_targets(k=self.lookahead)
        if self.mirror and self.phase >= 14:
            target[:, [0, 3]] *= -1
        return np.concatenate((obs, target.flatten()))

    # def sample_next_next_step(self):
    #     self.pitch_prob /= self.pitch_prob.sum()
    #     inds = self.np_random.choice(np.arange(self.pitch_sample_size), p=self.pitch_prob.reshape(-1), size=1, replace=False)
    #     pitch = np.pi/2 + self.pitch_samples[inds[0]]
    #     #print(self.r_prob)
    #     yaw = 0
    #     r = 0.35
    #     self.next_pitch = self.next_next_pitch
    #     self.next_yaw = self.next_next_yaw
    #     self.next_dr = np.copy(self.next_next_dr)
        
    #     self.next_next_pitch = pitch
    #     self.next_next_yaw = yaw
    #     self.next_next_dr = r
    #     self.set_next_next_step_location(self.next_next_pitch, self.next_next_yaw, self.next_next_dr)


    def sample_next_next_step(self):
        pairs = np.indices(dimensions=(self.yaw_sample_size,self.pitch_sample_size))
        #print("prob", np.round(self.yaw_pitch_prob, 2))
        self.yaw_pitch_prob /= self.yaw_pitch_prob.sum() 
        #print("prob sum", self.yaw_pitch_prob.sum())
        inds = self.np_random.choice(np.arange(self.yaw_sample_size*self.pitch_sample_size), p=self.yaw_pitch_prob.reshape(-1),size=1,replace=False)
        #print(inds)
        inds = pairs.reshape(2, self.yaw_sample_size*self.pitch_sample_size)[:, inds].squeeze()
        #print(np.round(self.yaw_pitch_prob, 2), inds)
        yaw = self.yaw_samples[inds[0]]
        pitch = self.pitch_samples[inds[1]] + np.pi / 2

        #continuous sampling
        pitch = self.np_random.uniform(self.pitch_samples[5-self.curriculum], self.pitch_samples[5+self.curriculum])
        pitch = pitch + np.pi/2
        
        self.next_pitch = self.next_next_pitch
        self.next_yaw = self.next_next_yaw
        self.next_dr = np.copy(self.next_next_dr)
        
        self.next_next_pitch = pitch
        self.next_next_yaw = yaw
        self.next_next_dr = 0.35#self.np_random.uniform(*self.r_range)
        self.set_next_next_step_location(self.next_next_pitch, self.next_next_yaw, self.next_next_dr)

    def sample_next_next_step_2(self):
        #print(np.round(self.yaw_pitch_prob, 2))
        pairs = np.indices(dimensions=(self.yaw_sample_size,self.pitch_sample_size, self.r_sample_size))
        #print("prob", np.round(self.yaw_pitch_prob, 2))
        self.yaw_pitch_r_prob /= self.yaw_pitch_r_prob.sum() 
        #print("prob sum", self.yaw_pitch_prob.sum())
        inds = self.np_random.choice(np.arange(self.yaw_sample_size*self.pitch_sample_size*self.r_sample_size), p=self.yaw_pitch_r_prob.reshape(-1),size=1,replace=False)
        #print(inds)
        inds = pairs.reshape(3, self.yaw_sample_size*self.pitch_sample_size*self.r_sample_size)[:, inds].squeeze()
        yaw = self.yaw_samples[inds[0]]
        pitch = self.pitch_samples[inds[1]] + np.pi / 2
        dr = self.r_samples[inds[2]]
        
        self.next_pitch = self.next_next_pitch
        self.next_yaw = self.next_next_yaw
        self.next_dr = np.copy(self.next_next_dr)
        
        self.next_next_pitch = pitch
        self.next_next_yaw = yaw
        self.next_next_dr = dr
        self.set_next_next_step_location(self.next_next_pitch, self.next_next_yaw, self.next_next_dr)

    def sample_next_next_step_3(self):
        #print(np.round(self.yaw_pitch_prob, 2))
        pairs = np.indices(dimensions=(self.yaw_sample_size,self.pitch_sample_size, self.r_sample_size, self.x_tilt_sample_size, self.y_tilt_sample_size))
        #print("prob", np.round(self.yaw_pitch_prob, 2))
        self.yaw_pitch_r_tilt_prob /= self.yaw_pitch_r_tilt_prob.sum() 
        #print("prob sum", self.yaw_pitch_prob.sum())
        inds = self.np_random.choice(np.arange(self.yaw_sample_size*self.pitch_sample_size*self.r_sample_size*self.x_tilt_sample_size*self.y_tilt_sample_size), p=self.yaw_pitch_r_tilt_prob.reshape(-1),size=1,replace=False)
        #print(inds)
        inds = pairs.reshape(5, self.yaw_sample_size*self.pitch_sample_size*self.r_sample_size*self.x_tilt_sample_size*self.y_tilt_sample_size)[:, inds].squeeze()
        yaw = self.yaw_samples[inds[0]]
        pitch = self.pitch_samples[inds[1]] + np.pi / 2
        dr = self.r_samples[inds[2]]
        x_tilt = self.x_tilt_samples[inds[3]]
        y_tilt = self.y_tilt_samples[inds[4]]
        
        self.next_pitch = self.next_next_pitch
        self.next_yaw = self.next_next_yaw
        self.next_dr = np.copy(self.next_next_dr)
        self.next_x_tilt = np.copy(self.next_next_x_tilt)
        self.next_y_tilt = np.copy(self.next_next_y_tilt)
        
        self.next_next_pitch = pitch
        self.next_next_yaw = yaw
        self.next_next_dr = dr
        self.next_next_x_tilt = x_tilt
        self.next_next_y_tilt = y_tilt
        self.set_next_next_step_location(self.next_next_pitch, self.next_next_yaw, self.next_next_dr, x_tilt=self.next_next_x_tilt, y_tilt=self.next_next_y_tilt)

    def create_temp_states(self):
        if self.update_terrain:
            temp_states = []
            for r in self.fake_r_samples:
                self.set_next_step_location(np.pi/2, 0, r)
                self.set_next_next_step_location(np.pi/2, 0, 0.35)
                temp_state = self.get_temp_state()
                temp_states.append(temp_state)
            self.set_next_step_location(self.next_pitch, self.next_yaw, self.next_dr)
            self.set_next_next_step_location(self.next_next_pitch, self.next_next_yaw, self.next_next_dr)
            ret = np.stack(temp_states)
        else:
            ret = self.temp_states
        return ret

    def create_temp_states_1(self):
        if self.update_terrain:
            temp_states = []
            for yaw in self.fake_yaw_samples:
                for pitch in self.fake_pitch_samples:
                    actual_pitch = np.pi/2 - pitch
                    self.set_next_step_location(actual_pitch, yaw, 0.4)
                    self.set_next_next_step_location(np.pi/2, 0, 0.35)
                    temp_state = self.get_temp_state()
                    temp_states.append(temp_state)
            self.set_next_step_location(self.next_pitch, self.next_yaw, self.next_dr)
            self.set_next_next_step_location(self.next_next_pitch, self.next_next_yaw, self.next_next_dr)
            ret = np.stack(temp_states)
        else:
            ret = self.temp_states
        return ret

    def create_temp_states_2(self):
        if self.update_terrain:
            temp_states = []
            for yaw in self.fake_yaw_samples:
                for pitch in self.fake_pitch_samples:
                    for r in self.fake_r_samples:
                        actual_pitch = np.pi/2 - pitch
                        self.set_next_step_location(actual_pitch, yaw, r)
                        self.set_next_next_step_location(np.pi/2, 0, 0.35)
                        temp_state = self.get_temp_state()
                        temp_states.append(temp_state)
            #print(self.fake_pitch_samples)
            self.set_next_step_location(self.next_pitch, self.next_yaw, self.next_dr)
            self.set_next_next_step_location(self.next_next_pitch, self.next_next_yaw, self.next_next_dr)
            ret = np.stack(temp_states)
        else:
            ret = self.temp_states
        return ret

    def create_temp_states_3(self):
        if self.update_terrain:
            temp_states = []
            for yaw in self.fake_yaw_samples:
                for pitch in self.fake_pitch_samples:
                    for r in self.fake_r_samples:
                        for x_tilt in self.x_tilt_samples:
                            for y_tilt in self.y_tilt_samples:
                                actual_pitch = np.pi/2 - pitch
                                self.set_next_step_location(actual_pitch, yaw, r, x_tilt=x_tilt, y_tilt=y_tilt)
                                self.set_next_next_step_location(np.pi/2, 0, 0.35)
                                temp_state = self.get_temp_state()
                                temp_states.append(temp_state)
            #print(self.fake_pitch_samples)
            self.set_next_step_location(self.next_pitch, self.next_yaw, self.next_dr, x_tilt=self.next_x_tilt, y_tilt=self.next_y_tilt)
            self.set_next_next_step_location(self.next_next_pitch, self.next_next_yaw, self.next_next_dr, x_tilt=self.next_next_x_tilt, y_tilt=self.next_next_y_tilt)
            ret = np.stack(temp_states)
        else:
            ret = self.temp_states
        return ret

    def update_sample_prob(self, sample_prob):
        self.pitch_prob = sample_prob
        return

    def update_sample_prob_1(self, sample_prob):
        self.yaw_pitch_prob = sample_prob
        return
        if self.update_terrain:
            self.yaw_pitch_prob = sample_prob
            self.update_terrain_info()
            self.avg_prob += sample_prob
            self.av_size += 1

    def update_sample_prob_2(self, sample_prob):
        self.yaw_pitch_r_prob = sample_prob
        return

    def update_sample_prob_3(self, sample_prob):
        self.yaw_pitch_r_tilt_prob = sample_prob
        return

    def update_curriculum_0(self, curriculum):
        self.curriculum = min(curriculum, self.max_curriculum)
        half_size = (self.sample_size-1)//2
        if self.curriculum >= half_size:
            self.curriculum = half_size
        self.pitch_prob *= 0
        prob = 1.0 / (self.curriculum * 2 + 1)
        window = slice(half_size-self.curriculum, half_size+self.curriculum+1)
        self.pitch_prob[window] = prob

    def update_curriculum(self, curriculum):
        # self.yaw_pitch_prob *= 0
        # self.yaw_pitch_prob[(self.yaw_sample_size-1)//2, (self.pitch_sample_size-1)//2] = 1
        self.curriculum = min(curriculum, self.max_curriculum)
        half_size = (self.sample_size-1)//2
        if self.curriculum >= half_size:
            self.curriculum = half_size
        self.yaw_pitch_prob *= 0
        prob = 1.0 / (self.curriculum * 2 + 1)**2
        window = slice(half_size-self.curriculum, half_size+self.curriculum+1)
        self.yaw_pitch_prob[window, window] = prob

    def update_curriculum_2(self, curriculum):
        # self.yaw_pitch_r_prob *= 0
        # self.yaw_pitch_r_prob[(self.yaw_sample_size-1)//2, (self.pitch_sample_size-1)//2, 0] = 1
        self.curriculum = min(curriculum, self.max_curriculum)
        half_size = (self.sample_size-1)//2
        if self.curriculum >= half_size:
            self.curriculum = half_size
        self.yaw_pitch_r_prob *= 0
        prob = 1.0 / (self.curriculum * 2 + 1)**3
        window = slice(half_size-self.curriculum, half_size+self.curriculum+1)
        self.yaw_pitch_r_prob[window, window, 0:curriculum*2+1] = prob

    def update_curriculum_3(self, curriculum):
        self.yaw_pitch_r_tilt_prob *= 0
        self.yaw_pitch_r_tilt_prob[(self.yaw_sample_size-1)//2, (self.pitch_sample_size-1)//2, 0, 2, 2] = 1
        return
        # self.curriculum = min(curriculum, self.max_curriculum)
        # half_size = (self.sample_size-1)//2
        # if self.curriculum >= half_size:
        #     self.curriculum = half_size
        # self.yaw_pitch_r_prob *= 0
        # prob = 1.0 / (self.curriculum * 2 + 1)**3
        # window = slice(half_size-self.curriculum, half_size+self.curriculum+1)
        # self.yaw_pitch_r_tilt_prob[window, window, 0:curriculum*2+1,] = prob

    def update_specialist(self, specialist):
        self.specialist = min(specialist, 5)
        prev_specialist = self.specialist - 1
        #print((self.specialist * 2 + 1)**2 - (prev_specialist*2+1)**2)
        half_size = (self.sample_size-1)//2
        if specialist == 0:
            prob = 1
        else:
            prob = 1.0 / ((self.specialist * 2 + 1)**2 - (prev_specialist*2+1)**2)
        window = slice(half_size-self.specialist, half_size+self.specialist+1)
        prev_window = slice(half_size-prev_specialist, half_size+prev_specialist+1)
        self.yaw_pitch_prob *= 0
        self.yaw_pitch_prob[window, window] = prob
        self.yaw_pitch_prob[prev_window, prev_window] = 0
        #print(np.round(self.yaw_pitch_prob, 2))


    def calc_potential(self):

        walk_target_theta = np.arctan2(
            self.walk_target[1] - self.sim.data.qpos[1],
            self.walk_target[0] - self.sim.data.qpos[0],
        )
        walk_target_delta = self.walk_target - self.sim.data.qpos[0:3]

        w, x, y, z = np.copy(self.sim.data.qpos[3:7])
        self.angle_to_target = (
            walk_target_theta - pybullet.getEulerFromQuaternion((x, y, z, w))[2]
        )
        #print("yaw", pybullet.getEulerFromQuaternion((x, y, z, w))[2], walk_target_theta, self.walk_target[1] - self.sim.data.qpos[1], self.walk_target[0] - self.sim.data.qpos[0])

        self.distance_to_target = (
            walk_target_delta[0] ** 2 + walk_target_delta[1] ** 2
        ) ** (1 / 2)

        self.linear_potential = -self.distance_to_target / (
            1.0 / 60
        )
        #self.angular_potential = np.cos(self.angle_to_target)
        self.angular_potential = -(self.angle_to_target)**2 / (1.0/60)
        #print(self.angle_to_target, self.angular_potential)

    def calc_progress_reward(self):
        old_linear_potential = self.linear_potential
        old_angular_potential = self.angular_potential
        self.calc_potential()
        linear_progress = self.linear_potential - old_linear_potential
        angular_progress = self.angular_potential - old_angular_potential
        #print(linear_progress, angular_progress)
        self.progress = linear_progress

    def calc_step_state(self):

        if not hasattr(self, "foot_body_ids"):
            return

        # +1 because 0 is floor
        target_step_id = (self.next_step_index % self.rendered_step_count) + 1
        step_pos = self.terrain_info[self.next_step_index, 0:3]

        self.target_reached = False
        self.step_bonus = 0
        for contact_id in range(self.data.ncon):

            id1 = self.sim.model.geom_bodyid[self.data.contact[contact_id].geom1]
            id2 = self.sim.model.geom_bodyid[self.data.contact[contact_id].geom2]
            contact_pos = self.sim.data.contact[contact_id].pos[:]

            object_id = min(id1, id2)
            robot_part_id = max(id1, id2)

            if robot_part_id in self.foot_body_ids and ((contact_pos - step_pos)**2).sum() < 0.01:# and object_id == target_step_id:
                self.target_reached = True
                contact_pos = self.data.contact[contact_id].pos
                delta = step_pos - contact_pos
                distance = (delta[0] ** 2 + delta[1] ** 2) ** (1 / 2)
                self.step_bonus = 20 * np.exp(-distance / self.step_radius)
                break

        if self.target_reached:
            self.target_reached_count += 1

            # Make target stationary for a bit
            if self.target_reached_count >= self.stop_frames:
                self.next_step_index += 1
                self.target_reached_count = 0
                # self.update_steps()

            # Prevent out of bound
            if self.next_step_index >= len(self.terrain_info):
                self.next_step_index -= 1

    def delta_to_k_targets(self, k=1):
        """ Return positions (relative to root) of target, and k-1 step after """
        targets = self.terrain_info[self.next_step_index : self.next_step_index + k]
        if len(targets) < k:
            # If running out of targets, repeat last target
            targets = np.concatenate(
                (targets, np.repeat(targets[[-1]], k - len(targets), axis=0))
            )

        self.walk_target = targets[[1], 0:3].mean(axis=0)

        deltas = targets[:, 0:3] - self.sim.data.qpos[0:3]
        target_thetas = np.arctan2(deltas[:, 1], deltas[:, 0])

        w, x, y, z = np.copy(self.sim.data.qpos[3:7])
        angle_to_targets = target_thetas - pybullet.getEulerFromQuaternion((x, y, z, w))[2]
        distance_to_targets = np.linalg.norm(deltas[:, 0:2], ord=2, axis=1)

        deltas = np.stack(
            (
                np.sin(angle_to_targets) * distance_to_targets,  # x
                np.cos(angle_to_targets) * distance_to_targets,  # y
                deltas[:, 2],  # z
                targets[:, 4],  # x_tilt
                targets[:, 5],  # y_tilt
            ),
            axis=1,
        )

        self.delta = deltas
        return deltas

    def set_next_next_step_location(self, pitch, yaw, dr, x_tilt=0, y_tilt=0):
        next_step_xyz = self.terrain_info[self.next_step_index]
        # if random:
        #     dr = self.np_random.uniform(*self.r_range)
        # else:
        #     dr = 0.35
        dr = dr
        base_phi = self.base_phi[self.next_step_index + 1]
        base_yaw = self.terrain_info[self.next_step_index, 3]  #not sure whether to do +1 for index

        dx = dr * np.sin(pitch) * np.cos(yaw + base_phi)
        # clip to prevent overlapping
        dx = np.sign(dx) * min(max(abs(dx), self.step_radius * 3.5), self.r_range[1])
        dy = dr * np.sin(pitch) * np.sin(yaw + base_phi)

        matrix = np.array([
            [np.cos(base_yaw), -np.sin(base_yaw)],
            [np.sin(base_yaw), np.cos(base_yaw)]
        ])

        dxy = np.dot(matrix, np.concatenate(([dx], [dy])))

        x = next_step_xyz[0] + dxy[0]
        y = next_step_xyz[1] + dxy[1]
        z = next_step_xyz[2] + dr * np.cos(pitch)

        self.terrain_info[self.next_step_index + 1, 0] = x
        self.terrain_info[self.next_step_index + 1, 1] = y
        self.terrain_info[self.next_step_index + 1, 2] = z
        self.terrain_info[self.next_step_index + 1, 3] = yaw + base_yaw
        self.terrain_info[self.next_step_index + 1, 4] = x_tilt
        self.terrain_info[self.next_step_index + 1, 5] = y_tilt

    def set_next_step_location(self, pitch, yaw, dr, x_tilt=0, y_tilt=0):
        next_step_xyz = self.terrain_info[self.next_step_index-1]
        dr = dr
        base_phi = self.base_phi[self.next_step_index]
        base_yaw = self.terrain_info[self.next_step_index, 3]

        dx = dr * np.sin(pitch) * np.cos(yaw + base_phi)
        # clip to prevent overlapping
        dx = np.sign(dx) * min(max(abs(dx), self.step_radius * 3.5), self.r_range[1])
        dy = dr * np.sin(pitch) * np.sin(yaw + base_phi)

        matrix = np.array([
            [np.cos(base_yaw), -np.sin(base_yaw)],
            [np.sin(base_yaw), np.cos(base_yaw)]
        ])

        dxy = np.dot(matrix, np.concatenate(([dx], [dy])))

        x = next_step_xyz[0] + dxy[0]
        y = next_step_xyz[1] + dxy[1]
        z = next_step_xyz[2] + dr * np.cos(pitch)

        self.terrain_info[self.next_step_index, 0] = x
        self.terrain_info[self.next_step_index, 1] = y
        self.terrain_info[self.next_step_index, 2] = z
        self.terrain_info[self.next_step_index, 3] = yaw + base_yaw
        self.terrain_info[self.next_step_index, 4] = x_tilt
        self.terrain_info[self.next_step_index, 5] = y_tilt


def euler2quat(z=0, y=0, x=0):
    z = z / 2.0
    y = y / 2.0
    x = x / 2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    result = np.array(
        [
            cx * cy * cz - sx * sy * sz,
            cx * sy * sz + cy * cz * sx,
            cx * cz * sy - sx * cy * sz,
            cx * cy * sz + sx * cz * sy,
        ]
    )
    if result[0] < 0:
        result = -result
    return result


def quaternion2euler(quaternion):
    w = quaternion[0]
    x = quaternion[1]
    y = quaternion[2]
    z = quaternion[3]
    ysqr = y * y
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.degrees(math.atan2(t0, t1))
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.degrees(math.atan2(t3, t4))
    result = np.zeros(3)
    result[0] = X * np.pi / 180
    result[1] = Y * np.pi / 180
    result[2] = Z * np.pi / 180
    return result


if __name__ == "__main__":
    env = CassieStepperEnv()
    env = CassieEnv()

    import torch
    from model import *

    Net = ActorCriticNet
    model = Net(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        hidden_layer=[256, 256],
        num_contact=2,
    )

    state_dict = torch.load(os.path.join(current_dir, "cassie_gym_seed8.pt"))
    model.load_state_dict(state_dict)

    obs = env.reset()

    while True:
        # env.step(env.action_space.sample() * 0)

        with torch.no_grad():
            mu = model.sample_best_actions(torch.from_numpy(obs).float().unsqueeze(0))

        action = mu.squeeze().numpy()
        obs, reward, done, _ = env.step(action)
        env.render()

        time.sleep(0.002)
