import rospy
import time
import numpy as np
import cv2
import sys
import os
import random
import math
from math import pi
import copy

import gymnasium as gym
from gymnasium import spaces
from gym_hydrone.envs import gazebo_env
from geometry_msgs.msg import Twist
from gym_hydrone.envs.respawnGoal import Respawn
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from sensor_msgs.msg import LaserScan
#from gym.utils import seeding
from cv_bridge import CvBridge, CvBridgeError

#import skimage as skimage
#from skimage import transform, color, exposure
#from skimage.transform import rotate
#from skimage.viewer import ImageViewer

TURTLE = ''
STATE_H, STATE_W = 100, 100

class HydroneEnv(gym.Env):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        #gazebo_env.GazeboEnv.__init__(self, "/home/hydrone/catkin_ws/src/hydrone_deep_rl_icra/hydrone_aerial_underwater_deep_rl/launch/hydrone_deep_rl.launch")

        rospy.init_node('gym')

        self.pub_cmd_vel = rospy.Publisher('/hydrone_aerial_underwater/gazebo/command/motor_speed', Actuators, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.env_stage = 1

        self.respawn_goal = Respawn()
        #self._seed()
        self.start_time = time.time()
        self.num_timesteps = 0
        self.initGoal = True
        
        self.collision_distance = 0.35
        self.goalbox_distance = 0.65

        self.min_alt = -5.
        self.max_alt = 5.
        
        self.min_range = 0.25
        self.max_range = 8.

        self.min_ang_vel = -0.25
        self.min_altitude_vel = -0.25
        self.min_linear_vel = -0.25

        self.max_ang_vel = 0.25
        self.max_altitude_vel = 0.25
        self.max_linear_vel = 0.25
        
        self.img_rows = STATE_H
        self.img_cols = STATE_W
        self.img_channels = 3

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(low = 0, high = 255, shape=(self.img_channels,self.img_rows,self.img_cols), dtype=int),
                "state": spaces.Box(low = -2**63, high = 2**63 - 2, shape=(13,), dtype=np.float32),
                "target": spaces.Box(low = -2**63, high = 2**63 - 2, shape=(3,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(low=0, high=800, shape=(4,), dtype=np.float32)
        
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_laser(self):
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/hydrone_aerial_underwater/scan', LaserScan, timeout=5)
            except:
                pass
        scan = np.asarray(data.ranges)
        scan[np.isnan(scan)] = self.min_range
        scan[np.isinf(scan)] = self.max_range
        
        return scan
        
    def _get_camera(self):
        image_data = None
        success=False
        cv_image = None
        while image_data is None :
            try:
                image_data = rospy.wait_for_message("/hydrone_aerial_underwater/camera/rgb/image_raw", Image, timeout=5)
                cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
                #temporal fix, check image is not corrupted
                
            except:
                pass
        #cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("Current_O_pixel", cv_image)
        #cv2.waitKey(3)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
        cv_image = cv_image.reshape(self.img_channels, cv_image.shape[0], cv_image.shape[1])
        return cv_image
        
    def _get_state_and_heading(self):
        state = np.zeros((13,))
        odom = None
        while odom is None:
            try:
                odom = rospy.wait_for_message('/hydrone_aerial_underwater/ground_truth/odometry', Odometry, timeout=5)
            except:
                pass
        state[0:3] = [odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z]
        state[3:6] = [odom.twist.twist.linear.x,odom.twist.twist.linear.y,odom.twist.twist.linear.z]
        state[6:10] = [odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,odom.pose.pose.orientation.z,odom.pose.pose.orientation.w]
        state[10:13] = [odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z]
        
        orientation_list = state[6:10] 
        position = state[0:3]
        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        
        yaw_angle = math.atan2(self.goal_y - position[1], self.goal_x - position[0])
        pitch_angle = math.atan2(self.goal_z - position[2], self.goal_x - position[0])

        heading = np.array([0., 0., 0.])
        heading[0] = yaw_angle - yaw
        heading[1] = pitch_angle - pitch
        for i in range(2):
            if heading[i] > math.pi:
                heading[i] -= 2 * math.pi

            elif heading[i] < -math.pi:
                heading[i] += 2 * math.pi
        
        goal_distance = math.sqrt((self.goal_x - position[0])**2 + (self.goal_y - position[1])**2 + (self.goal_z - position[2])**2)
        heading[2] = goal_distance

        return state, heading
        
    def _get_obs(self):
        state, heading = self._get_state_and_heading()
        return {"image": self._get_camera(), "state": state, "target": heading}

    def _get_info(self):
        time_info = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.start_time))
        time_info += '-' + str(self.num_timesteps)
        return {'time_info': time_info}

    def _goal_position(self):
        if self.env_stage == 1:
                    targets = np.asarray([np.random.uniform((-3.5, -3.5, 0.05), (3.5, 3.5, 3.5)) for _ in range(1)])
        if self.env_stage == 2:
            targets = np.asarray([np.random.uniform((-3.5, -3.5, 0.5), (3.5, 3.5, 3.5)) for _ in range(1)])
            if ((targets[0,0] < -1.6 and targets[0,0] > -2.9) or (targets[0,0] > 1.6 and targets[0,0] < 2.9) 
                and (targets[0,1] < -1.6 and targets[0,1] > -2.9) or (targets[0,1] > 1.6 and targets[0,1] < 2.9)):
                targets = np.asarray([np.random.uniform((-3.5, -3.5, 0.5), (3.5, 3.5, 3.5)) for _ in range(1)])

        if self.env_stage == 3:
            targets = np.asarray([np.random.uniform((-4.0, -4.0, 0.5), (4.0, 4.0, 3.5)) for _ in range(1)])
            if not(((targets[0,0] > -3.15 and targets[0,0] < -4.0) and (targets[0,1] > 2.25 and targets[0,1] < -4.3)) or
                   ((targets[0,0] > 4.3 and targets[0,0] < 0.6) and (targets[0,1] > -2.25 and targets[0,1] > -4.3)) or
                   ((targets[0,0] > 4.3 and targets[0,0] < 3.6) and (targets[0,1] > 4.3 and targets[0,1] < 4.1)) or
                   ((targets[0,0] > 1.6 and targets[0,0] < 0.0) and (targets[0,1] > 1.5 and targets[0,1] < -4.3)) or
                   ((targets[0,0] > 1.3 and targets[0,0] < -1.44) and (targets[0,1] > 4.3 and targets[0,1] < 3.4))):
                targets = np.asarray([np.random.uniform((-4.0, -4.0, 0.5), (4.0, 4.0, 3.5)) for _ in range(1)])
        
        return targets
           
    def reset(self):
        #rospy.loginfo("Reset!! ")
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        targets = self._goal_position()[0]
        self.goal_x = targets[0]
        self.goal_y = targets[1]
        self.goal_z = targets[2]        
        
        if self.initGoal:
            self.respawn_goal.setPosition(targets)
            self.initGoal = False
            time.sleep(1)
        else:
            self.respawn_goal.setPosition(targets, delete=True)
        observation = self._get_obs()
        '''rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")'''
        
        return observation, self._get_info()
    
    
    def step(self, action):
        #rospy.loginfo("Step!! ")
        self.num_timesteps += 1
        '''rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")'''
        
        observation = self._get_obs()
        lidar_distances = self._get_laser()
        
        reward = 0.
        terminated = False
        
        if min(lidar_distances) < self.collision_distance:
            # print(f'{time_info}: Collision!!')
            reward = -1.
            #terminated = True
            #rospy.loginfo("Collision!!")

        if (observation['state'][2] < self.min_alt or observation['state'][2] > self.max_alt):
            reward = -1.
            #terminated = True
            #rospy.loginfo("too high!!")

        if observation['target'][2] < self.goalbox_distance:
            reward = 1.
            #rospy.loginfo("Goal!! ")
            #terminated = True
        
        '''vel_cmd = Twist()
        vel_cmd.angular.z = np.clip(action[0], self.min_ang_vel, self.max_ang_vel)
        vel_cmd.linear.x = np.clip(action[1], self.min_linear_vel, self.max_linear_vel)
        vel_cmd.linear.z = np.clip(action[2], self.min_altitude_vel, self.max_altitude_vel)'''
        vel_cmd = Actuators()
        vel_cmd.angular_velocities = np.clip(action, 0, 800)
        self.pub_cmd_vel.publish(vel_cmd)
                
        '''rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")'''
            
        
        return observation, reward, terminated, False, self._get_info()

