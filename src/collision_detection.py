#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import math
import numpy as np 

import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler

 
from delta_msgs.msg import (EgoStateEstimateArray,
                            OncomingVehicleTrajectoryArray,
                            CollisionDetection)
from obstacles import collisionChecking

EGO_VEHICLE_FRAME = 'ego_vehicle'

class CollisionDetectionClass:
    def __init__(self, publisher):
        self.publisher = publisher
        self.ego_traj = []
        self.oncoming_trajs = {}
        self.npc_bb = np.array([5.5, 2.5])
        self.ego_bb = np.array([6.0, 3.0])
        self.traj_len=20
        self.probability = 0.0

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
        for i in range(self.traj_len):
            for key, value in self.oncoming_trajs.items():
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
                    self.probability = np.clip(self.probability + 0.1, 0.0, 1.0)
                    sys.stdout.write("\r*******Collision Vehicle ID: %02d in %.1f secs with %.1f probability*******\t" % (
                        key, i / 10.0, self.probability))
                    sys.stdout.flush()
                    self.publish_collision_msg(key, i/10.0, value[i])
                    return
        self.probability = np.clip(self.probability - 0.3, 0.0, 1.0)


    def publish_collision_msg(self, track_id, ttc, state):
        msg = CollisionDetection()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = EGO_VEHICLE_FRAME
        msg.time_to_impact = rospy.Duration.from_sec(ttc)
        msg.probability = self.probability
        msg.track_id = track_id
        msg.state.x = state[0]
        msg.state.y = state[1]
        msg.state.vx = state[2]
        msg.state.vy = state[3]
        self.publisher['collision'].publish(msg)
        
            
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

    collisionobj  = CollisionDetectionClass(publishers)
    
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