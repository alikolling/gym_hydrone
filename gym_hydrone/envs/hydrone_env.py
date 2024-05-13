import math
import time

import gymnasium as gym
import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose
from gymnasium import spaces
from mav_msgs.msg import Actuators
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion

from gym_hydrone.envs.respawnGoal import Respawn


class HydroneEnv(gym.Env):

    def __init__(self):
        rospy.init_node("gym")

        self.pub_cmd_vel = rospy.Publisher(
            "/haubentaucher/gazebo/command/motor_speed", Actuators, queue_size=1
        )
        self.reset_srv = rospy.ServiceProxy("gazebo/set_model_state", SetModelState)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.env_stage = 1

        self.respawn_goal = Respawn()
        # self._seed()
        self.start_time = time.time()
        self.num_timesteps = 0
        self.initGoal = True
        self.last_action = [0.0, 0.0, 0.0, 0.0]
        self.collision_distance = 0.35
        self.goalbox_distance = 0.05

        self.min_alt = -5.0
        self.max_alt = 5.0

        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=-(2**63), high=2**63 - 2, shape=(20,), dtype=np.float32
                ),
                "target": spaces.Box(
                    low=-(2**63), high=2**63 - 2, shape=(3,), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Box(low=0, high=1800, shape=(4,), dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_state_and_heading(self, last_action):
        state = np.zeros((13,))
        odom = None
        while odom is None:
            try:
                odom = rospy.wait_for_message(
                    "/haubentaucher/ground_truth/odometry", Odometry, timeout=5
                )
            except:
                pass
        state[0:3] = [
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z,
        ]
        state[3:6] = [
            odom.twist.twist.linear.x,
            odom.twist.twist.linear.y,
            odom.twist.twist.linear.z,
        ]
        state[6:10] = [
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w,
        ]
        state[10:13] = [
            odom.twist.twist.angular.x,
            odom.twist.twist.angular.y,
            odom.twist.twist.angular.z,
        ]
        state = np.concatenate([state, last_action])
        orientation_list = state[6:10]
        position = state[0:3]
        roll, pitch, yaw = euler_from_quaternion(orientation_list)

        yaw_angle = math.atan2(self.goal_y - position[1], self.goal_x - position[0])
        pitch_angle = math.atan2(self.goal_z - position[2], self.goal_x - position[0])

        heading = np.array([0.0, 0.0, 0.0])
        heading[0] = yaw_angle - yaw
        heading[1] = pitch_angle - pitch
        for i in range(2):
            if heading[i] > math.pi:
                heading[i] -= 2 * math.pi

            elif heading[i] < -math.pi:
                heading[i] += 2 * math.pi

        goal_distance = math.sqrt(
            (self.goal_x - position[0]) ** 2
            + (self.goal_y - position[1]) ** 2
            + (self.goal_z - position[2]) ** 2
        )
        heading[2] = goal_distance

        return state, heading

    def _get_obs(self):
        state, heading = self._get_state_and_heading(self.last_action)
        return {"state": state, "target": heading}

    def _get_info(self):
        time_info = time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - self.start_time)
        )
        time_info += "-" + str(self.num_timesteps)
        return {"time_info": time_info}

    def _random_position(self):
        if self.env_stage == 1:
            targets = np.asarray(
                [
                    np.random.uniform((-1.5, -1.5, 1.50), (1.5, 1.5, 2.5))
                    for _ in range(1)
                ]
            )
        if self.env_stage == 2:
            targets = np.asarray(
                [
                    np.random.uniform((-3.5, -3.5, 0.5), (3.5, 3.5, 3.5))
                    for _ in range(1)
                ]
            )
            if (
                (targets[0, 0] < -1.6 and targets[0, 0] > -2.9)
                or (targets[0, 0] > 1.6 and targets[0, 0] < 2.9)
                and (targets[0, 1] < -1.6 and targets[0, 1] > -2.9)
                or (targets[0, 1] > 1.6 and targets[0, 1] < 2.9)
            ):
                targets = np.asarray(
                    [
                        np.random.uniform((-3.5, -3.5, 0.5), (3.5, 3.5, 3.5))
                        for _ in range(1)
                    ]
                )

        if self.env_stage == 3:
            targets = np.asarray(
                [
                    np.random.uniform((-4.0, -4.0, 0.5), (4.0, 4.0, 3.5))
                    for _ in range(1)
                ]
            )
            if not (
                (
                    (targets[0, 0] > -3.15 and targets[0, 0] < -4.0)
                    and (targets[0, 1] > 2.25 and targets[0, 1] < -4.3)
                )
                or (
                    (targets[0, 0] > 4.3 and targets[0, 0] < 0.6)
                    and (targets[0, 1] > -2.25 and targets[0, 1] > -4.3)
                )
                or (
                    (targets[0, 0] > 4.3 and targets[0, 0] < 3.6)
                    and (targets[0, 1] > 4.3 and targets[0, 1] < 4.1)
                )
                or (
                    (targets[0, 0] > 1.6 and targets[0, 0] < 0.0)
                    and (targets[0, 1] > 1.5 and targets[0, 1] < -4.3)
                )
                or (
                    (targets[0, 0] > 1.3 and targets[0, 0] < -1.44)
                    and (targets[0, 1] > 4.3 and targets[0, 1] < 3.4)
                )
            ):
                targets = np.asarray(
                    [
                        np.random.uniform((-4.0, -4.0, 0.5), (4.0, 4.0, 3.5))
                        for _ in range(1)
                    ]
                )

        return targets

    def reset(self):

        # Unpause simulation to make observation
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            # resp_pause = pause.call()
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        targets = [0.0, 0.0, 2.0]
        self.goal_x = targets[0]
        self.goal_y = targets[1]
        self.goal_z = targets[2]

        self.initial_vehicle_position = [0.0, 0.0, 2.0]

        vel_cmd = Actuators()
        vel_cmd.angular_velocities = [0.0, 0.0, 0.0, 0.0]
        self.pub_cmd_vel.publish(vel_cmd)

        reset_state = ModelState()
        reset_state.model_name = "haubentaucher"
        pose = Pose()
        pose.position.x = self.initial_vehicle_position[0]
        pose.position.y = self.initial_vehicle_position[1]
        pose.position.z = self.initial_vehicle_position[2]
        reset_state.pose = pose

        self.reset_srv(reset_state)

        reset_state = ModelState()
        reset_state.model_name = "goal_box"
        pose = Pose()
        pose.position.x = self.goal_x
        pose.position.y = self.goal_y
        pose.position.z = self.goal_z
        reset_state.pose = pose

        self.reset_srv(reset_state)

        observation = self._get_obs()

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            # resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        return observation, self._get_info()

    def _get_reward(self, observation):
        reward_col = -1.0
        reward_target = 1.0
        roll, pitch, yaw = euler_from_quaternion(observation["state"][6:10])

        if (
            roll > math.pi / 4
            or roll < -math.pi / 4
            or pitch > math.pi / 4
            or pitch < -math.pi / 4
        ):
            self.initial_vehicle_position = [0.0, 0.0, 2.0]

            vel_cmd = Actuators()
            vel_cmd.angular_velocities = [0.0, 0.0, 0.0, 0.0]
            self.pub_cmd_vel.publish(vel_cmd)
            reset_state = ModelState()
            reset_state.model_name = "haubentaucher"
            pose = Pose()
            pose.position.x = self.initial_vehicle_position[0]
            pose.position.y = self.initial_vehicle_position[1]
            pose.position.z = self.initial_vehicle_position[2]
            reset_state.pose = pose
            self.reset_srv(reset_state)
            print(f"Reward flip: {reward_col}", end="\r", flush=True)
            return reward_col

        if observation["target"][2] < self.goalbox_distance:

            # rospy.loginfo("Goal!! ")
            # terminated = True
            print(f"Reward: {reward_target}", end="\r", flush=True)
            return reward_target

        reward_dist = max(
            0.0,
            1
            - np.linalg.norm(
                np.asarray([self.goal_x, self.goal_y, self.goal_z])
                - np.asarray(observation["state"][0:3])
            )
            - math.sqrt((self.goal_z - observation["state"][2]) ** 2),
        )

        if observation["state"][2] < 0.5:
            reward_dist -= 0.5

        print(f"Reward: {reward_dist}", end="\r", flush=True)

        return reward_dist

    def step(self, action):
        # rospy.loginfo("Step!! ")
        self.num_timesteps += 1
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        observation = self._get_obs()

        reward = self._get_reward(observation)

        terminated = False

        vel_cmd = Actuators()
        vel_cmd.angular_velocities = np.clip(action, 0, 1800)
        self.pub_cmd_vel.publish(vel_cmd)

        self.last_action = action

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            # resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        return observation, reward, terminated, False, self._get_info()
