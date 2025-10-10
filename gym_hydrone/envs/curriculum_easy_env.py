import math
import time

import gymnasium as gym
import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, DeleteModel, SpawnModel
from geometry_msgs.msg import Pose
from gymnasium import spaces
from mav_msgs.msg import Actuators
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped
from sensor_msgs.msg import LaserScan
from typing import Optional



class HydroneCurriculumEasyEnv(gym.Env):

    def __init__(self):
        rospy.init_node("gym")
        self.pub_aerial_cmd_vel = rospy.Publisher(
            "/haubentaucher/command/motor_speed", Actuators, queue_size=1
        )
        self.pub_thruster00 = rospy.Publisher(
            "/haubentaucher/thrusters/0/input", FloatStamped, queue_size=1
        )
        self.pub_thruster01 = rospy.Publisher(
            "/haubentaucher/thrusters/1/input", FloatStamped, queue_size=1
        )
        self.pub_thruster02 = rospy.Publisher(
            "/haubentaucher/thrusters/2/input", FloatStamped, queue_size=1
        )
        self.reset_srv = rospy.ServiceProxy(
            "gazebo/set_model_state", SetModelState)
        self.delete_model_srv = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.spawn_model_srv = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        
        self.env_stage = 0  # curriculum stage (0 to 3)
        self.deleted_obstacles = False
        
        self.start_time = time.time()
        self.num_timesteps = 0
        self.goal = np.asarray([0.0, 0.0, 2.0])
        self.goal_orientation = np.zeros(
            3,
        )
        self.initial_vehicle_position = None
        self.initial_vehicle_orientation = None
        self.action_base = [1500, 1500, 1500, 1500, 100.0, 0.0, 0.0]
        self.last_action = [1500, 1500, 1500, 1500, 0.0, 0.0, 0.0]
        self.collision_distance = 0.5
        self.goalbox_distance = 0.1
        self.quat_distance = 0.1

        self.min_alt = -5.0
        self.max_alt = 5.0
        self.min_range = 0.5
        self.max_range = 8.

        self.observation_space = spaces.Box(
            low=-(2**63), high=2**63 - 2, shape=(23,), dtype=np.float32
        )
        low_action = np.asarray([0.0, 0.0, 0.0, 0.0, -300.0, -300.0, -300.0])
        high_action = np.asarray([1800.0, 1800.0, 1800.0, 1800.0, 300.0, 300.0, 300.0]) 
        self.action_space = spaces.Box(
            low=low_action, high=high_action, shape=(7,), dtype=np.float32)

    def _get_state_and_heading(self):
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
        
        return state
        
    def _get_laser(self):
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/haubentaucher/scan', LaserScan, timeout=5)
            except:
                pass
        scan = np.asarray(data.ranges)
        scan[np.isnan(scan)] = self.min_range
        scan[np.isinf(scan)] = self.max_range
        
        return scan
    
    def _get_sonar(self):
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/haubentaucher/sonar', LaserScan, timeout=5)
            except:
                pass
        sonar = np.asarray(data.ranges)
        sonar[np.isnan(sonar)] = self.min_range
        sonar[np.isinf(sonar)] = self.max_range
        
        return sonar

    def _get_obs(self):
        state = self._get_state_and_heading()
        #scan = self._get_laser() if state[2] >= 0 else np.zeros((20,)) 
        #sonar = self._get_sonar() if state[2] < 0 else np.zeros((20,))
        obs = np.concatenate([state, self.last_action])#, scan, sonar])
        
        return obs

    def _get_info(self):
        time_info = time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - self.start_time)
        )
        time_info += "-" + str(self.num_timesteps)
        return {"time_info": time_info}

    def _random_position(self):
        if self.env_stage == 0:
            return np.random.uniform(low=(0.0, 0.0, -1.5), high=(0.0, 0.0, 1.5))  # navigate to goal air and water
        elif self.env_stage == 1:
            return np.random.uniform(low=(-1.0, -1.0, -1.5), high=(1.0, 1.0, 1.5))  # navigate to goal air and water
        elif self.env_stage == 2:
            return np.random.uniform(low=(-4.0, -4.0, -4.5), high=(4.0, 4.0, 4.5))  # navigate to goal air and water
        elif self.env_stage == 3:
            obstacles = [
                (2.0, 2.0, -2.5), (-2.0, -2.0, -2.5), (2.0, -2.0, -2.5), (-2.0, 2.0, -2.5),
                (-2.0, 6.0, -2.5), (-6.0, 2.0, -2.5), (6.0, 2.0, -2.5), (6.0, -2.0, -2.5),
                (-6.0, -2.0, -2.5), (6.0, 6.0, -2.5), (-6.0, 6.0, -2.5), (6.0, -6.0, -2.5),
                (-6.0, -6.0, -2.5), (-2.0, -6.0, -2.5), (2.0, -6.0, -2.5), (2.0, 6.0, -2.5)
            ]
            obstacle_radius = 1.0
            forbidden_zone_margin = 0.5

            while True:
                target = np.random.uniform(
                    low=(-6.0, -6.0, -4.5),
                    high=(6.0, 6.0, 10.)
                )
                collision = False
                for obs in obstacles:
                    if np.linalg.norm(target[:2] - np.array(obs[:2])) < obstacle_radius + forbidden_zone_margin:
                        collision = True
                        break
                if not collision:
                    return target

    def delete_obstacles(self):
        for i in range(1, 18):  # obstacle_1 to obstacle_17
            model_name = f"obstacle_{i}"
            try:
                self.delete_model_srv(model_name)
            except rospy.ServiceException as e:
                rospy.logwarn(f"Failed to delete {model_name}: {e}")
        self.deleted_obstacles = True

    def spawn_obstacles(self):
        sdf_path = "/home/hydrone/.gazebo/models/obstacle_1/model.sdf"
        with open(sdf_path, "r") as f:
            model_xml = f.read()

        obstacle_poses = [
            (2.0, 2.0, -2.5),
            (-2.0, -2.0, -2.5),
            (2.0, -2.0, -2.5),
            (-2.0, 2.0, -2.5),
            (-2.0, 6.0, -2.5),
            (-6.0, 2.0, -2.5),
            (6.0, 2.0, -2.5),
            (6.0, -2.0, -2.5),
            (-6.0, -2.0, -2.5),
            (6.0, 6.0, -2.5),
            (-6.0, 6.0, -2.5),
            (6.0, -6.0, -2.5),
            (-6.0, -6.0, -2.5),
            (-2.0, -6.0, -2.5),
            (2.0, -6.0, -2.5),
            (2.0, 6.0, -2.5),
            (2.0, 6.0, -2.5),
        ]

        for i, (x, y, z) in enumerate(obstacle_poses):
            model_name = f"obstacle_{i+1}"
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            pose.orientation.w = 1.0  # No rotation
            try:
                self.spawn_model_srv(model_name, model_xml, "", pose, "world")
            except rospy.ServiceException as e:
                rospy.logwarn(f"Failed to spawn {model_name}: {e}")
        self.deleted_obstacles = False
        
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
            if self.initial_vehicle_position[2] >= 0:
                vel_cmd = Actuators()
                vel_cmd.angular_velocities = self.action_base[:4]
                self.pub_aerial_cmd_vel.publish(vel_cmd)
                
            elif self.initial_vehicle_position[2] < 0:
                vel_cmd = Actuators()
                vel_cmd.angular_velocities = [0., 0., 0., 0.]
                self.pub_aerial_cmd_vel.publish(vel_cmd)
            
            
            thrusters_thrust = self.action_base[4:]
                
            thrusters_thrust00_msg = FloatStamped()
            thrusters_thrust00_msg.data = thrusters_thrust[0]
            self.pub_thruster00.publish(thrusters_thrust00_msg)

            thrusters_thrust01_msg = FloatStamped()
            thrusters_thrust01_msg.data = thrusters_thrust[1]
            self.pub_thruster01.publish(thrusters_thrust01_msg)

            thrusters_thrust02_msg = FloatStamped()
            thrusters_thrust02_msg.data = thrusters_thrust[2]
            self.pub_thruster02.publish(thrusters_thrust02_msg)
            
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
        #definitions
        terminated = False
        success = False
        reward_col = -10.0
        reward_target = 10.0
        
        #variables
        roll, pitch, yaw = euler_from_quaternion(observation[3:7])

        def euler_to_quaternion(roll, pitch, yaw):
            rotation = Rotation.from_euler(
                "xyz", [roll, pitch, yaw], degrees=False)
            quaternion = rotation.as_quat(canonical=True)
            return quaternion

        dist = np.linalg.norm(np.asarray(observation[0:3]) - self.goal)
        attitude_penalty = abs(roll) + abs(pitch)
        quat_sim = np.dot(observation[3:7], euler_to_quaternion(
            self.goal_orientation[0],
            self.goal_orientation[1],
            self.goal_orientation[2],
        ))
        quat_penalty = 1 - abs(quat_sim)
        ang_vel_err = np.linalg.norm(observation[10:13])
        lin_vel_err = np.linalg.norm(observation[3:6])
        
        if observation[2] > 0.0:#aerial
            punish_action = np.linalg.norm(np.zeros(3) - observation[-3:]) / 300
            base_action = np.linalg.norm(self.action_base[:4] - observation[-7:-3])/1800
        elif observation[2] <= 0.0:#underwater
            punish_action = np.linalg.norm(np.zeros(4) - observation[-7:-3])/1800
            base_action = np.linalg.norm(self.action_base[4:] - observation[-3:])/300
        
        reward = 1.0
        reward -= 0.3 * dist
        reward -= 0.3 * quat_penalty
        reward -= 0.1 * lin_vel_err
        reward -= 0.1 * ang_vel_err
        reward -= 0.05 * base_action
        reward -= 0.05 * punish_action
        reward -= 0.1 * attitude_penalty
        reward -= 0.01  # small time cost per step
        reward = max(-1.0, reward)
        
        #conditions
        if (
            roll > math.pi / 2
            or roll < -math.pi / 2
            or pitch > math.pi / 2
            or pitch < -math.pi / 2
            or np.any(observation[:2] > 6.0)
            or np.any(observation[:2] < -6.0)
            or (observation[3] > 12.0)
            or (observation[3] < -6.0)
        ):

            terminated = True
            return reward_col, terminated, success

        if dist < self.goalbox_distance and quat_penalty < self.quat_distance:
            success = True

            return reward_target, terminated, success
        
        if self.env_stage == 0:
            terminated = dist > 3. # if too far, terminate
            
            return reward, terminated, success

        elif self.env_stage == 1:
            terminated = dist > 3.  # if too far, terminate
            
            return reward, terminated, success

        elif self.env_stage in [2, 3, 4]:
            return reward, terminated, success


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.env_stage = options['stage'] if options else 0
        if self.env_stage < 1 and not self.deleted_obstacles:
            self.delete_obstacles()
        elif self.env_stage == 1 and self.deleted_obstacles:
            self.spawn_obstacles()
        
        # Unpause simulation to make observation
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            # resp_pause = pause.call()
            self.unpause()
        except rospy.ServiceException:
            print("/gazebo/unpause_physics service call failed")
        
        
        self.initial_vehicle_position=self._random_position()
        self.initial_vehicle_orientation=self._random_orientation()
        self.goal=self._random_position()
        
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

        rotors_vel = action[:4]
        rotors_vel = np.clip(rotors_vel, 0, 1800)
        rotors_vel_msg = Actuators()
        rotors_vel_msg.angular_velocities = rotors_vel
        self.pub_aerial_cmd_vel.publish(rotors_vel_msg)
        
        thrusters_thrust = action[4:]
        thrusters_thrust = np.clip(thrusters_thrust, -300, 300)
        
        thrusters_thrust00_msg = FloatStamped()
        thrusters_thrust00_msg.data = thrusters_thrust[0]
        self.pub_thruster00.publish(thrusters_thrust00_msg)

        thrusters_thrust01_msg = FloatStamped()
        thrusters_thrust01_msg.data = thrusters_thrust[1]
        self.pub_thruster01.publish(thrusters_thrust01_msg)

        thrusters_thrust02_msg = FloatStamped()
        thrusters_thrust02_msg.data = thrusters_thrust[2]
        self.pub_thruster02.publish(thrusters_thrust02_msg)

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
