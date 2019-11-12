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

DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi


class CassieEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.init_x = 0
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
        self.time = 0
        self.phase = 0
        self.counter = 0
        self.time_limit = 400
        self.max_phase = 28

        # used for rendering
        self.controller_frameskip = 30

        # fake
        self.foot_body_ids = np.zeros(2).astype(np.int)

        mujoco_env.MujocoEnv.__init__(self, "cassie.xml", 1)
        utils.EzPickle.__init__(self)

        self.foot_body_ids = np.array(
            [
                self.model._body_name2id["left-foot"],
                self.model._body_name2id["right-foot"],
            ]
        )

    def get_state(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        ref_pos, ref_vel = self.get_kin_next_state()
        pos_index = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 20, 21, 22, 23, 28, 29, 30, 34]
        )
        vel_index = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 18, 19, 20, 21, 25, 26, 27, 31]
        )

        foot_xyzs = self.sim.data.body_xpos[self.foot_body_ids]
        height = self.sim.data.qpos[2] - np.min(foot_xyzs[:, 2])

        state = np.concatenate(
            [qpos[pos_index], qvel[vel_index], ref_pos[pos_index], ref_vel[vel_index]]
        )
        state[0] = 0
        state[1] = height
        return state

    def step_simulation(self, action):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        pos_index = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        vel_index = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]
        ref_pos, ref_vel = self.get_kin_next_state()
        target = action + ref_pos[pos_index]
        control = self.P * (target - qpos[pos_index]) - self.D * qvel[vel_index]
        self.do_simulation(control, self.frame_skip)

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
        done = not (height > 0.4 and height < 3.0) or self.time >= self.time_limit
        reward = self.compute_reward()

        if reward < 0.3:
            done = True

        return state, reward, done, {}

    def get_kin_state(self):
        pose = np.copy(self.trajectory.qpos[self.phase * 2 * 30])
        pose[0] += (
            self.trajectory.qpos[1681, 0] - self.trajectory.qpos[0, 0]
        ) * self.counter
        pose[1] = 0
        vel = np.copy(self.trajectory.qvel[self.phase * 2 * 30])
        return pose, vel

    def get_kin_next_state(self):
        phase = (self.phase + 1) % self.max_phase
        pose = np.copy(self.trajectory.qpos[phase * 2 * 30])
        vel = np.copy(self.trajectory.qvel[phase * 2 * 30])
        pose[0] += (
            self.trajectory.qpos[1681, 0] - self.trajectory.qpos[0, 0]
        ) * self.counter
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

    def reset(self, height_offset=0):
        self.phase = np.random.randint(0, 27)
        self.time = 0
        self.counter = 0
        qpos, qvel = self.get_kin_state()
        qpos[2] += height_offset
        self.set_state(qpos, qvel)
        foot_xyzs = self.sim.data.body_xpos[self.foot_body_ids]
        self.init_x = np.mean(foot_xyzs[:, 0])
        return self.get_state()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 0.3
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20

    # def render(self, mode="human"):
    #     super().render(mode)
    #     self.viewer.cam.lookat[:] = self.sim.data.qpos[0:3]


class CassieStepperEnv(CassieEnv):
    def __init__(self, render=False):
        self.lookahead = 2
        self.next_step_index = 0
        self.target_reached_count = 0
        self.stop_frames = 0
        self.step_bonus = 0

        self.n_steps = 50
        self.pitch_limit = 20
        self.yaw_limit = 0
        self.tilt_limit = 0

        self.step_radius = 0.12  # defined in xml
        self.rendered_step_count = 4  # defined in xml
        self.initial_height = 20

        self.terrain_info = np.zeros((self.n_steps, 6))
        self.linear_potential = 0
        self.walk_target = np.zeros(3)

        from gym.envs.mujoco.model import ActorCriticNet

        Net = ActorCriticNet
        self.base_model = Net(
            80,  # observation dim
            10,  # action dim
            hidden_layer=[256, 256],
            num_contact=2,
        )

        state_dict = torch.load(os.path.join(current_dir, "cassie_gym_seed8.pt"))
        self.base_model.load_state_dict(state_dict)

        super().__init__()

        self.all_contact_geom_ids = {
            self.sim.model.geom_bodyid[self.model._geom_name2id[key]]: key
            for key in ["floor", "step1", "step2", "step3", "step4"]
        }

    def randomize_terrain(self):
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

            self.model.body_pos[index + 1, :] = pos[:]
            self.model.body_quat[index + 1, :] = euler2quat(phi, y_tilt, x_tilt)

    def generate_step_placements(
        self,
        n_steps=50,
        min_gap=0.35,
        max_gap=0.45,
        yaw_limit=30,
        pitch_limit=25,
        tilt_limit=10,
    ):

        r_range = np.array([min_gap, max_gap])
        y_range = np.array([-yaw_limit, yaw_limit]) * DEG2RAD
        p_range = np.array([90 - pitch_limit, 90 + pitch_limit]) * DEG2RAD
        t_range = np.array([-tilt_limit, tilt_limit]) * DEG2RAD

        dr = np.random.uniform(*r_range, size=n_steps)
        dphi = np.random.uniform(*y_range, size=n_steps)
        dtheta = np.random.uniform(*p_range, size=n_steps)

        # make first step slightly further to accommodate different starting poses
        dr[0] = self.init_x
        dphi[0] = 0.0
        dtheta[0] = np.pi / 2

        dr[1] = 0.4
        dphi[1] = 0.0
        dtheta[1] = np.pi / 2

        x_tilt = np.random.uniform(*t_range, size=n_steps)
        y_tilt = np.random.uniform(*t_range, size=n_steps)

        dphi = np.cumsum(dphi)

        x_ = dr * np.sin(dtheta) * np.cos(dphi)
        y_ = dr * np.sin(dtheta) * np.sin(dphi)
        z_ = dr * np.cos(dtheta)

        # Prevent steps from overlapping
        np.clip(x_[2:], a_min=self.step_radius * 3, a_max=max_gap, out=x_[2:])

        x = np.cumsum(x_)
        y = np.cumsum(y_)
        z = np.cumsum(z_) + self.initial_height

        # because xyz is the centre of box, need to account for height
        z[:] -= 0.098

        min_z = self.step_radius * np.sin(self.tilt_limit * DEG2RAD) + 0.01
        np.clip(z, a_min=min_z - 0.098, a_max=None, out=z)

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
            self.model.body_pos[oldest + 1, :] = pos[:]
            self.model.body_quat[oldest + 1, :] = euler2quat(phi, y_tilt, x_tilt)

    def reset(self):

        self.next_step_index = 0
        self.target_reached_count = 0

        obs = super().reset(height_offset=self.initial_height)
        self.randomize_terrain()

        self.targets = self.delta_to_k_targets(k=self.lookahead)

        self.calc_potential()

        return np.concatenate((obs, self.targets.flatten()))

    def step(self, action):
        obs = torch.from_numpy(super().get_state()).float()

        with torch.no_grad():
            base_action = self.base_model.sample_best_actions(obs).squeeze().numpy()

        obs, _, done, _ = super().step(action + base_action)

        # check if target changed
        cur_step_index = self.next_step_index

        self.calc_progress_reward()
        self.calc_step_state()

        if cur_step_index != self.next_step_index:
            self.calc_potential()

        self.targets = self.delta_to_k_targets(k=self.lookahead)

        return (
            np.concatenate((obs, self.targets.flatten())),
            self.progress + self.step_bonus,
            done,
            {},
        )

    def calc_potential(self):

        walk_target_theta = np.arctan2(
            self.walk_target[1] - self.sim.data.qpos[1],
            self.walk_target[0] - self.sim.data.qpos[0],
        )
        walk_target_delta = self.walk_target - self.sim.data.qpos[0:3]

        self.angle_to_target = (
            walk_target_theta - quaternion2euler(self.sim.data.qpos[3:7])[2]
        )

        self.distance_to_target = (
            walk_target_delta[0] ** 2 + walk_target_delta[1] ** 2
        ) ** (1 / 2)

        self.linear_potential = -self.distance_to_target / (
            1.0 / self.controller_frameskip
        )

    def calc_progress_reward(self):
        old_linear_potential = self.linear_potential
        self.calc_potential()
        linear_progress = self.linear_potential - old_linear_potential
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

            object_id = min(id1, id2)
            robot_part_id = max(id1, id2)

            if robot_part_id in self.foot_body_ids and object_id == target_step_id:
                self.target_reached = True
                contact_pos = self.data.contact[contact_id].pos
                delta = step_pos - contact_pos
                distance = (delta[0] ** 2 + delta[1] ** 2) ** (1 / 2)
                self.step_bonus = 10 * np.exp(-distance / self.step_radius)
                break

        if self.target_reached:
            self.target_reached_count += 1

            # Make target stationary for a bit
            if self.target_reached_count >= self.stop_frames:
                self.next_step_index += 1
                self.target_reached_count = 0
                self.update_steps()

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

        # self.walk_target = targets[[1], 0:3].mean(axis=0)

        deltas = targets[:, 0:3] - self.sim.data.qpos[0:3]
        target_thetas = np.arctan2(deltas[:, 1], deltas[:, 0])

        angle_to_targets = target_thetas - quaternion2euler(self.sim.data.qpos[3:7])[2]
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

        # Normalize targets x,y to between -1 and +1 using softsign
        # deltas[:, 0:2] /= 1 + np.abs(deltas[:, 0:2])

        return deltas


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
