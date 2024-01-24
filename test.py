import warnings
from collections import deque, defaultdict
from typing import Any, NamedTuple
import numpy as np
import gymnasium as gym
import time
import gym_hydrone
import os
import rospy
warnings.filterwarnings("ignore", category=DeprecationWarning) 

time.sleep(5)
os.environ['ROS_MASTER_URI'] = "http://localhost:{}/".format(11310 + 1)
#rospy.init_node('Potter_Circuit_Simple-v0'.replace('-', '_') + "_w{}".format(1))
env = gym.make('gym_hydrone/hydrone-v0')
time.sleep(5)

observation, info = env.reset()
for _ in range(100):
    action = [1.0, 1.0, 1.0]
    observation, reward, terminated, _, info = env.step(action)
    print(reward)
    if terminated:
        observation = env.reset()

env.close()
