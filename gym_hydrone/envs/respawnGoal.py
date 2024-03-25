#!/usr/bin/env python3
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert, Eduardo #

import rospy
import time
import os
import rospkg
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose


class Respawn:
    def __init__(self):
        rospack = rospkg.RosPack()
        self.modelPath = os.path.join(rospack.get_path('hydrone_aerial_underwater_deep_rl'), 'models/goal_box/model.sdf')
        # print(self.modelPath)
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
        self.modelName = 'goal'
        self.check_model = False
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)

    def checkModel(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.check_model = True

    def respawnModel(self, goal_position):
        goal_pose = Pose()
        goal_pose.position.x = goal_position[0]
        goal_pose.position.y = goal_position[1]
        goal_pose.position.z = goal_position[2]
        while True:
            if not self.check_model:
                rospy.wait_for_service('gazebo/spawn_sdf_model')
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                spawn_model_prox(self.modelName, self.model, 'robotos_name_space', goal_pose, "world")
                break
            else:
                pass

    def deleteModel(self):
        while True:
            if self.check_model:
                rospy.wait_for_service('gazebo/delete_model')
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                del_model_prox(self.modelName)
                break
            else:
                pass


    def setPosition(self, goal_position, delete=False):
        if delete:
            self.deleteModel()

        time.sleep(0.5)
        self.respawnModel(goal_position)
