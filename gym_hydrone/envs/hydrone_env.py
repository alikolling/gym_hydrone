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
from scipy.spatial.transform import Rotation
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler


class HydroneEnv(gym.Env):

    def __init__(self):
        rospy.init_node("gym")

        self.pub_cmd_vel = rospy.Publisher(
            "/haubentaucher/gazebo/command/motor_speed", Actuators, queue_size=1
        )
        self.reset_srv = rospy.ServiceProxy(
            "gazebo/set_model_state", SetModelState)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)

        self.start_time = time.time()
        self.num_timesteps = 0
        self.goal = np.asarray([0.0, 0.0, 2.0])
        self.goal_orientation = np.zeros(
            3,
        )
        self.initial_vehicle_position = None
        self.initial_vehicle_orientation = None
        self.action_base = [1000, 1000, 1000, 1000]
        self.last_action = [0.0, 0.0, 0.0, 0.0]
        self.collision_distance = 0.35
        self.goalbox_distance = 0.05

        self.min_alt = -5.0
        self.max_alt = 5.0

        self.observation_space = spaces.Box(
            low=-(2**63), high=2**63 - 2, shape=(20,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=0, high=1800, shape=(4,), dtype=np.float32)

    def _get_state_and_heading(self, last_action):
        state = np.zeros((13,))
        odom = None
        while odom is None:
            try:
                odom = rospy.wait_for_message(
                    "/haubentaucher/odometry_sensor1/odometry", Odometry, timeout=5
                )
            except rospy.ServiceException:
                pass
        state[0:3] = [
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z,
        ]
        state[3:7] = [
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w,
        ]
        state[7:10] = [
            odom.twist.twist.linear.x,
            odom.twist.twist.linear.y,
            odom.twist.twist.linear.z,
        ]
        state[10:13] = [
            odom.twist.twist.angular.x,
            odom.twist.twist.angular.y,
            odom.twist.twist.angular.z,
        ]

        orientation_list = state[3:7]
        position = state[0:3]
        _, pitch, yaw = euler_from_quaternion(orientation_list)

        yaw_angle = math.atan2(
            self.goal[1] - position[1], self.goal[0] - position[0])
        pitch_angle = math.atan2(
            self.goal[2] - position[2], self.goal[0] - position[0])

        heading = np.array([0.0, 0.0, 0.0])
        heading[0] = yaw_angle - yaw
        heading[1] = pitch_angle - pitch
        for i in range(2):
            if heading[i] > math.pi:
                heading[i] -= 2 * math.pi

            elif heading[i] < -math.pi:
                heading[i] += 2 * math.pi

        goal_distance = math.sqrt(
            (self.goal[0] - position[0]) ** 2
            + (self.goal[1] - position[1]) ** 2
            + (self.goal[2] - position[2]) ** 2
        )
        heading[2] = goal_distance

        state = np.concatenate([state, heading])
        obs = np.concatenate([state, last_action])
        return obs

    def _get_obs(self):
        obs = self._get_state_and_heading(self.last_action)
        return obs

    def _get_info(self):
        time_info = time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - self.start_time)
        )
        time_info += "-" + str(self.num_timesteps)
        return {"time_info": time_info}

    def _random_position(self):
        targets = np.random.uniform(
            (-100.5, -100.5, 1.50), (100.5, 100.5, 12.5))

        return targets

    def _random_orientation(self):
        def euler_to_quaternion(roll, pitch, yaw):
            rotation = Rotation.from_euler(
                "xyz", [roll, pitch, yaw], degrees=False)
            quaternion = rotation.as_quat(canonical=True)
            return quaternion

        # Generate random Euler angles within specified ranges
        roll = np.random.uniform(-math.pi / 6, math.pi / 6)
        pitch = np.random.uniform(-math.pi / 6, math.pi / 6)
        yaw = np.random.uniform(0, 2 * math.pi)

        # Convert Euler angles to quaternion
        quaternion = euler_to_quaternion(roll, pitch, yaw)
        return quaternion

    def _reset_state(self, model_name: str):
        if model_name == "haubentaucher":

            vel_cmd = Actuators()
            vel_cmd.angular_velocities = self.last_action
            self.pub_cmd_vel.publish(vel_cmd)
            reset_state = ModelState()
            reset_state.model_name = "haubentaucher"
            pose = Pose()
            pose.position.x = self.initial_vehicle_position[0]
            pose.position.y = self.initial_vehicle_position[1]
            pose.position.z = self.initial_vehicle_position[2]
            pose.orientation.x = self.initial_vehicle_orientation[0]
            pose.orientation.y = self.initial_vehicle_orientation[1]
            pose.orientation.z = self.initial_vehicle_orientation[2]
            pose.orientation.w = self.initial_vehicle_orientation[3]
            reset_state.pose = pose
            self.reset_srv(reset_state)

        elif model_name == "goal_box":
            reset_state = ModelState()
            reset_state.model_name = "goal_box"
            pose = Pose()
            pose.position.x = self.goal[0]
            pose.position.y = self.goal[1]
            pose.position.z = self.goal[2]
            reset_state.pose = pose

            self.reset_srv(reset_state)
        else:
            pass

    def _get_reward(self, observation):
        terminated = False
        success = False
        reward_col = -10.0
        reward_target = 1.0
        roll, pitch, yaw = euler_from_quaternion(observation[3:7])
        if (
            roll > math.pi / 2
            or roll < -math.pi / 2
            or pitch > math.pi / 2
            or pitch < -math.pi / 2
        ):
            self._reset_state("haubentaucher")
            print(f"Reward flip: {reward_col}", end="\r", flush=True)
            terminated = True
            return reward_col, terminated, success

        """ if (
            roll > math.pi / 4
            or roll < -math.pi / 4
            or pitch > math.pi / 4
            or pitch < -math.pi / 4
        ):
            print(f"Reward flip: {reward_col}", end="\r", flush=True)
            return reward_col """

        def quaternion_distance(q1, q2):
            r1 = Rotation.from_quat(q1)
            r2 = Rotation.from_quat(q2)

            relative_rotation = r1.inv() * r2

            return relative_rotation.magnitude()

        def euler_to_quaternion(roll, pitch, yaw):
            rotation = Rotation.from_euler(
                "xyz", [roll, pitch, yaw], degrees=False)
            quaternion = rotation.as_quat(canonical=True)
            return quaternion

        dist = np.linalg.norm(self.goal - np.asarray(observation[0:3]))
        quat_dist = quaternion_distance(
                observation[3:7],
                euler_to_quaternion(
                    self.goal_orientation[0],
                    self.goal_orientation[1],
                    self.goal_orientation[2],
                ))
        
        if dist < self.goalbox_distance and quat_dist < 0.1:
            success = True
            return reward_target, terminated, success

        reward_dist=max(
            0.0,
            1.0
            - 0.4 * dist - 0.45 * quat_dist
            - 0.05 * np.linalg.norm(self.action_base - observation[-4:])/1800,
        )

        print(f"Reward dist: {reward_dist}", end="\r", flush=True)

        return reward_dist, terminated, success

    def reset(self):

        # Unpause simulation to make observation
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            # resp_pause = pause.call()
            self.unpause()
        except rospy.ServiceException:
            print("/gazebo/unpause_physics service call failed")

        self.initial_vehicle_position=self._random_position()
        self.initial_vehicle_orientation=self._random_orientation()
        self.goal=self.initial_vehicle_position
        roll, pitch, yaw=euler_from_quaternion(
            self.initial_vehicle_orientation)
        self.goal_orientation[2]=yaw

        self._reset_state("haubentaucher")
        self._reset_state("goal_box")

        observation=self._get_obs()

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            # resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException:
            print("/gazebo/pause_physics service call failed")

        return observation, self._get_info()

    def step(self, action):
        # rospy.loginfo("Step!! ")
        terminated=False
        self.num_timesteps += 1
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException:
            print("/gazebo/unpause_physics service call failed")

        action=np.clip(action, 0, 1800)
        vel_cmd=Actuators()
        vel_cmd.angular_velocities=action
        self.pub_cmd_vel.publish(vel_cmd)

        observation=self._get_obs()

        reward, terminated, success=self._get_reward(observation)

        info = self._get_info()
        info['terminated'] = terminated
        info['success'] = success
        self.last_action=action

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            # resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException:
            print("/gazebo/pause_physics service call failed")

        return observation, reward, terminated, False, info
