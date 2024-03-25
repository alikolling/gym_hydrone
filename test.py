import warnings
from collections import deque, defaultdict
from typing import Any, NamedTuple
import numpy as np
import gymnasium as gym
import time
import gym_hydrone
from gym_hydrone.wrappers import gymnasium_wrapper
import os
import rospy
warnings.filterwarnings("ignore", category=DeprecationWarning) 

time.sleep(5)

os.environ['ROS_MASTER_URI'] = "http://localhost:{}/".format(11310 + 1)



env = gymnasium_wrapper.GymnasiumWrapper(gym.make('gym_hydrone/hydrone-v0'))

time.sleep(5)
observation = env.reset()
print(observation)
for _ in range(100):
    action = [1000.0, 1000.0, 1000.0, 1000.0]
    timestep = env.step(action)
    print(timestep)
    #if terminated:
observation = env.reset()
for _ in range(100):
    action = [1000.0, 1000.0, 1000.0, 1000.0]
    timestep = env.step(action)
    print(timestep)
env.close()
