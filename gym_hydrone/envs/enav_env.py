import math
import time

import gymnasium as gym
import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose, Vector3, Quaternion, Transform
from gymnasium import spaces
from mav_msgs.msg import Actuators
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped
from sensor_msgs.msg import LaserScan
from typing import Optional
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Twist
from std_msgs.msg import Header

class HydroneNavEasyEnv(gym.Env):

    def __init__(self):
        rospy.init_node("gym")

        # Replace motor speed publishers with trajectory publisher
        self.pub_trajectory = rospy.Publisher(
            "/haubentaucher/command/trajectory", MultiDOFJointTrajectory, queue_size=1
        )
        
        # Remove individual thruster publishers since we're using trajectory commands now
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
        self.current_position = None
        self.current_orientation = None
        self.num_thrusters = 3
        # thruster publishers
        self.pub_thruster = []
        for i in range(self.num_thrusters):
            topic = f"/haubentaucher/thrusters/{i}/input"
            pub = rospy.Publisher(topic, FloatStamped, queue_size=1)
            self.pub_thruster.append(pub)
        
        # thruster geometry (in body frame)
        self.thruster_positions = [
            np.array([0.12, 0.0005, -0.1]),
            np.array([-0.025, -0.205500, 0.08500]),
            np.array([-0.025, 0.205500, 0.08500]),
        ]
        self.thruster_axes = [
            np.array([0, 0, 1.0]),
            np.array([0, 0, 1.0]),
            np.array([0, 0, 1.0]),
        ]
        self.mass = 6.737
        self.inertia = np.diag([0.02953, 0.2303, 0.1604])  # approximate diagonal
        
        self.uuv_lin_Kp = np.array([2.0, 2.0, 8.0])
        self.uuv_lin_Kd = np.array([1.5, 1.5, 3.0])

        self.uuv_ang_Kp = np.array([0.3, 0.3, 0.3])
        self.uuv_ang_Kd = np.array([0.1, 0.1, 0.1])


        # water surface z threshold
        self.water_level_z = 0.0

        # Change action base to represent linear, angular velocities and altitude
        # [linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]
        self.current_lin_vel = [0.0, 0.0, 0.0]
        self.current_ang_vel = [0.0, 0.0, 0.0]
        self.action_base = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.last_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.collision_distance = 0.5
        self.goalbox_distance = 0.25

        self.min_alt = -5.0
        self.max_alt = 5.0
        self.min_range = 0.55
        self.max_range = 8.
        
        self.observation_space = spaces.Box(
            low=-(2**63), high=2**63 - 2, shape=(63,), dtype=np.float32
        )
        
        # Update action space for linear and angular velocities
        # [linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]
        low_action = np.asarray([-5.0, -5.0, -5.0, -1.0, -1.0, -1.0])  # Reduced angular velocities
        high_action = np.asarray([5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
        self.action_space = spaces.Box(
            low=low_action, high=high_action, shape=(6,), dtype=np.float32)

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
                
        self.current_position = [ odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z, ]
        self.current_orientation = [ odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w, ]        
        self.current_lin_vel = [ odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z, ]
        self.current_ang_vel = [ odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z, ]
        
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
        scan = self._get_laser() if state[2] >= 0 else np.zeros((20,)) 
        sonar = self._get_sonar() if state[2] < 0 else np.zeros((20,))
        # Update observation to use 6-dimensional action instead of 7
        obs = np.concatenate([state, self.last_action, sonar, scan])
        
        return obs

    def _get_info(self):
        time_info = time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - self.start_time)
        )
        time_info += "-" + str(self.num_timesteps)
        return {"time_info": time_info}

    def _random_position(self):
        # For simplicity, using basic random position generation
        # You might want to add obstacle avoidance logic here
        return np.random.uniform(
            low=(0.0, 0.0, 2.0),
            high=(0.0, 0.0, 2.5)
        )

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
            # Publish zero velocity command on reset
            
            
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
            #self._publish_trajectory_command([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

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
    
    def _compute_wrench_from_velocity(self, action):
        # action = [v_x, v_y, v_z, ω_x, ω_y, ω_z]
        v_cur = np.array(self.current_lin_vel)
        w_cur = np.array(self.current_ang_vel)

        v_err = action[0:3] - v_cur
        a_cmd = self.uuv_lin_Kp * v_err - self.uuv_lin_Kd * v_cur
        F_des = self.mass * a_cmd

        w_err = action[3:6] - w_cur
        alpha_cmd = self.uuv_ang_Kp * w_err - self.uuv_ang_Kd * w_cur
        tau_des = self.inertia.dot(alpha_cmd)

        return F_des, tau_des

    def _allocate_thrusters(self, F_des, tau_des):
        # number of thrusters = N
        N = len(self.thruster_positions)
        B = np.zeros((6, N))
        for i in range(N):
            ai = self.thruster_axes[i]
            ri = self.thruster_positions[i]
            B[0:3, i] = ai
            B[3:6, i] = np.cross(ri, ai)
        vec = np.concatenate([F_des, tau_des])
        # least squares solution
        f_vec, *_ = np.linalg.lstsq(B, vec, rcond=None)
        return f_vec

    def _publish_thrusters(self, thr_vals):
        thr_vals = np.clip(thr_vals, a_min=[-300.0, -300.0, -300.0], a_max=[300.0, 300.0, 300.0])
        for i, f in enumerate(thr_vals):
            msg = FloatStamped()
            msg.data = float(f)
            self.pub_thruster[i].publish(msg)
        
    def is_underwater(self):
        # if current position below water level
        return self.current_position[2] < self.water_level_z
    
    def _publish_trajectory_command(self, action):
        """Publish trajectory command with linear and angular velocities"""
        trajectory_msg = MultiDOFJointTrajectory()
        
        # Set header
        trajectory_msg.header = Header()
        trajectory_msg.header.seq = 0
        trajectory_msg.header.stamp = rospy.Time.now()
        trajectory_msg.header.frame_id = ''
        
        # Set joint names (empty as per your example)
        trajectory_msg.joint_names = ['']
        
        # Create trajectory point
        point = MultiDOFJointTrajectoryPoint()
        
        # Set transform (identity/zero)
        self.current_position = self.current_position if self.current_position else self.initial_vehicle_position 
        self.current_orientation = self.current_orientation if self.current_orientation else self.initial_vehicle_orientation
        transform = Transform()
        transform.translation = Vector3(self.current_position[0], self.current_position[1], self.current_position[2]) 
        transform.rotation = Quaternion(self.current_orientation[0], self.current_orientation[1], self.current_orientation[2], self.current_orientation[3])
        point.transforms = [transform]
        
        # Set velocities from action
        # action: [linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]
        twist = Twist()
        twist.linear = Vector3(action[0], action[1], action[2])
        twist.angular = Vector3(action[3], action[4], action[5])
        point.velocities = [twist]
        
        # Set accelerations (zero for now)
        #accel_twist = Twist()
        #accel_twist.linear = Vector3(0.0, 0.0, 0.0)
        #accel_twist.angular = Vector3(0.0, 0.0, 0.0)
        #point.accelerations = [accel_twist]
        
        # Set time from start
        point.time_from_start = rospy.Duration(0.1)  # 100ms
        
        trajectory_msg.points = [point]
        
        self.pub_trajectory.publish(trajectory_msg)

    def _get_reward(self, observation):
        terminated = False
        success = False
        reward_col = -1.0
        reward_target = 1.0
        roll, pitch, yaw = euler_from_quaternion(observation[3:7])
        if (
            roll > math.pi / 4
            or roll < -math.pi / 4
            or pitch > math.pi / 4
            or pitch < -math.pi / 4
            #or min(observation[20:]) < self.collision_distance
            #or np.any(observation[:3] > 6.0)
            #or np.any(observation[:3] < -6.0)
        ):
            self._reset_state("haubentaucher")
            terminated = True
            return reward_col, terminated, success

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
        ang_vel_err = np.linalg.norm(observation[10:13])
        lin_vel_err = np.linalg.norm(observation[3:6])
        
        if dist < self.goalbox_distance:
            success = True
            return reward_target, terminated, success
        
        # Simplified action penalty for trajectory commands
        # Penalize large velocities to encourage smooth control
        action_penalty = np.linalg.norm(self.last_action) / 10.0
        
        reward_dist = max(
            -1.0,
            1.0
            - 0.35 * dist
            - 0.4 * quat_dist
            - 0.1 * action_penalty
            - 0.05 * lin_vel_err
            - 0.05 * ang_vel_err,
        )

        return reward_dist, terminated, success

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        # Unpause simulation to make observation
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException:
            print("/gazebo/unpause_physics service call failed")

        self.initial_vehicle_position = self._random_position()
        self.initial_vehicle_orientation = np.zeros((4,))
        self.initial_vehicle_orientation[3] = 1
        self.goal = self._random_position()

        roll, pitch, yaw = euler_from_quaternion(
            self.initial_vehicle_orientation)
        self.goal_orientation[2] = yaw

        self._reset_state("haubentaucher")
        self._reset_state("goal_box")

        observation = self._get_obs()

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException:
            print("/gazebo/pause_physics service call failed")

        return observation, self._get_info()

    def step(self, action):
        terminated = False
        self.num_timesteps += 1
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException:
            print("/gazebo/unpause_physics service call failed")

        # Clip the action to the allowed range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        

        if self.is_underwater():
            # compute F_des and τ_des from action and current state
            F_des, tau_des = self._compute_wrench_from_velocity(action)
            thr_vals = self._allocate_thrusters(F_des, tau_des)
            self._publish_thrusters(thr_vals)
            #self._publish_trajectory_command([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            # Publish trajectory command instead of motor speeds
            self._publish_trajectory_command(action)
            #self._publish_thrusters([0.0, 0.0, 0.0])

        observation = self._get_obs()

        reward, terminated, success = self._get_reward(observation)

        info = self._get_info()
        info['terminated'] = terminated
        info['success'] = success
        self.last_action = action

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException:
            print("/gazebo/pause_physics service call failed")

        return observation, reward, terminated, False, info
