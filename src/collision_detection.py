#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import math
import pdb
import numpy as np 

import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from visualization_msgs.msg import Marker

 
from delta_msgs.msg import (EgoStateEstimateArray,
                            OncomingVehicleTrajectoryArray,
                            CollisionDetection)
from obstacles import collisionChecking

EGO_VEHICLE_FRAME = 'ego_vehicle'

class CollisionDetectionClass:
    def __init__(self, publisher, collision_prob_threshold):
        self.publisher = publisher
        self.ego_traj = []
        self.oncoming_trajs = {}
        self.npc_bb = np.array([5.0, 2.0])
        self.ego_bb = np.array([6.0, 3.0])
        self.traj_len=30
        self.probability = {}
        self.collision_prob_threshold = collision_prob_threshold

    def ego_prediction_callback(self, msg):
        self.ego_traj = []
        for i in range(len(msg.states)):
            explicit_quat = [msg.states[i].pose.orientation.x, \
                             msg.states[i].pose.orientation.y, \
                             msg.states[i].pose.orientation.z, \
                             msg.states[i].pose.orientation.w]
            self.ego_traj.append(np.array([msg.states[i].pose.position.x,
                                           msg.states[i].pose.position.y,
                                           euler_from_quaternion(explicit_quat)[2]]))

    def oncoming_prediction_callback(self, msg):
        self.oncoming_trajs = {}
        for i in range(len(msg.trajectory)):
            traj = []
            for j in range(len(msg.trajectory[i].state)):
                traj.append(np.array([msg.trajectory[i].state[j].x,
                                      msg.trajectory[i].state[j].y,
                                      msg.trajectory[i].state[j].vx, 
                                      msg.trajectory[i].state[j].vy]))
            self.oncoming_trajs[msg.trajectory[i].track_id] = traj

    def run(self):
        try:
            if len(self.ego_traj)!=0 and len(self.oncoming_trajs)!=0:
                self.collision_check()
        except IndexError:
            pass

    def collision_check(self):
        for key, value in self.oncoming_trajs.items():
            collision = False
            for i in range(self.traj_len):
                orientation = self.find_orientation(value, i)
                other_obj = [np.array([value[i][0], 
                                       value[i][1], 
                                       orientation]), 
                                       self.npc_bb]
                ego_obj = [np.array([self.ego_traj[i][0], 
                                     self.ego_traj[i][1], 
                                     self.ego_traj[i][2]]), 
                                     self.ego_bb]
                if collisionChecking(ego_obj, other_obj):
                    # if collision, increase the probability of collision for that id
                    collision = True
                    if(value[i][2] > 0):
                        c_time = 0.001
                    else:
                        c_time = 0.015
                    if key not in self.probability:
                        self.probability[key] = np.clip(10*c_time + c_time*self.traj_len/(i+1), 0.0, 1.0)
                    else: 
                        self.probability[key] = np.clip(self.probability[key] + 10*c_time + c_time*self.traj_len/(i+1), 0.0, 1.0)
                    
                    # print on screen
                    sys.stdout.write("\r********* Collision Vehicle ID: %02d in %.2f secs with %.2f probability %.2f velocity  *********\t" % (
                        key, i / 10.0, self.probability[key], value[i][2]))
                    sys.stdout.flush()
                    
                    self.publish_collision_msg(key, i/10.0, value[i], self.probability[key])
                    
                    if self.probability[key] > self.collision_prob_threshold:
                        self.publish_marker_msg(value[i], [3.0, 3.0, 3.0], frame_id=EGO_VEHICLE_FRAME, color=[1.0, 0.0, 0.0])
                    print(self.probability)
                    break
            if key in self.probability and collision is False:
                self.probability[key] = np.clip(self.probability[key] - 0.3, 0.0, 1.0)
        # for key in self.probability:
        #     self.probability[key] = np.clip(self.probability[key] - 0.3, 0.0, 1.0)


    def publish_collision_msg(self, track_id, ttc, state, probability):
        msg = CollisionDetection()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = EGO_VEHICLE_FRAME
        msg.time_to_impact = rospy.Duration.from_sec(ttc)
        msg.probability = probability
        msg.track_id = track_id
        msg.state.x = state[0]
        msg.state.y = state[1]
        msg.state.vx = state[2]
        msg.state.vy = state[3]
        self.publisher['collision'].publish(msg)

    def publish_marker_msg(self, state, scale, frame_id='/map', marker_id=0,
        duration=0.5, color=[1.0, 1.0, 1.0]):
        """ 
        Helper function for generating visualization markers
        
        Args:
            trajectory (array-like): (n, 2) array-like trajectory data
            frame_id (str): ROS TF frame id
            marker_id (int): Integer identifying the trajectory
            duration (rospy.Duration): How long the trajectory will be displayed for
            color (list): List of color floats from 0 to 1 [r,g,b]
        
        Returns: 
            Marker: A trajectory marker message which can be published to RViz
        """
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = frame_id
        marker.id = marker_id

        marker.type = marker.CUBE
        marker.text = str(marker_id)
        marker.action = marker.ADD
        marker.scale.x = scale[0]
        marker.scale.y = scale[1]
        marker.scale.z = scale[2]
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration(duration)
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = state[0]
        marker.pose.position.y = state[1]
        marker.pose.position.z = 0.5
        self.publisher['visualization'].publish(marker)
        
            
    def find_orientation(self, trajectory, index):
        if index == 0:
            first = 0
            second = 1
        elif index == len(trajectory)-1:
            first = len(trajectory)-2
            second = len(trajectory)-1
        else:
            first = index
            second = index+1
        y = trajectory[second][1] - trajectory[first][1]
        x = trajectory[second][0] - trajectory[first][0]
        return math.atan2(y,x)


def main():
    rospy.init_node('collision_detection_node', anonymous=True)

    publishers = {}
    publishers['collision'] = rospy.Publisher("/delta/prediction/collision", CollisionDetection, queue_size=5)
    publishers['visualization'] = rospy.Publisher("/delta/prediction/collision/visualization", Marker, queue_size=5)

    collision_prob_threshold = rospy.get_param("collision_threshold", 0.5)

    collisionobj  = CollisionDetectionClass(publishers, collision_prob_threshold)
    
    rospy.Subscriber('/delta/prediction/ego_vehicle/trajectory', EgoStateEstimateArray, collisionobj.ego_prediction_callback)
    rospy.Subscriber('/delta/prediction/oncoming_vehicle/trajectory', OncomingVehicleTrajectoryArray, collisionobj.oncoming_prediction_callback)

    r = rospy.Rate(10)

    try:
        while not rospy.is_shutdown():
            collisionobj.run()
            r.sleep()
        # window_err = np.mean(np.asarray(collisionobj.RMSE_window))
    except rospy.ROSInterruptException:
        rospy.loginfo('Shutting down')

if __name__ == "__main__":
    main()