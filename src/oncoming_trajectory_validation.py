#!/usr/bin/env python
# -*- coding: utf-8 -*-

#########################################################################
#####                         Delta Autonomy                        #####
#####                   Written By: Karmesh Yadav                   #####
#####                         Date: 05/11/19                        #####
#########################################################################

import os
import numpy as np 
import json
import rospy
import pdb

from delta_msgs.msg import TrackArray

class OncomingVehicleTrajectoryValidation:
    def __init__(self, folder, dt, T,  min_tracking_time=2.1):
        self.folder = folder
        self.min_tracking_time = min_tracking_time
        self.dt = dt
        self.T = T
        self.time_range = np.arange(0,self.T+0.1,0.1)

        self.data = {}

        if not os.path.isdir(folder):
            os.mkdir(self.folder)

    def tracker_gt_callback(self, track_msg):
        timestamp = track_msg.header.stamp.to_sec()
        for i in range(len(track_msg.tracks)):

            current_data = np.array([timestamp,
                                    track_msg.tracks[i].track_id,
                                    track_msg.tracks[i].x,
                                    track_msg.tracks[i].y,
                                    track_msg.tracks[i].vx,
                                    track_msg.tracks[i].vy])

            if track_msg.tracks[i].track_id in self.data:
                last_data = self.data[track_msg.tracks[i].track_id]
                
                self.data[track_msg.tracks[i].track_id] = np.hstack((last_data, current_data))
            else:
                self.data[track_msg.tracks[i].track_id] = current_data

    def save(self, file_name):
        self.convert_data()
        file_name = os.path.join(self.folder, file_name + ".txt")
        with open(file_name, 'w') as outfile:
            json.dump(self.data, outfile)

    def convert_data(self):
        for track_id, data in self.data.items():
            self.data[track_id] = list(data)

    def load(self, file_name):
        file_name = os.path.join(self.folder, file_name + ".txt")
        with open(file_name, 'r') as infile:
            data = json.load(infile)
        self.data = data


    def filter_data(self):
        new_data = {}
        for track_id, data in self.data.items():
            data = np.array(data).reshape(-1, 6)
            if (data[-1,0] - data[0,0]) < self.min_tracking_time:
                print("Deleting New Track Id: {}".format(track_id))
            else:
                new_data[int(track_id)] = data
        self.data = new_data
                
    def get_gt_trajectory(self, track_id, time):
        if track_id in self.data:
            gt_traj = self.data[track_id]
            gt_traj = gt_traj[gt_traj[:, 0] > time - self.dt]
            gt_traj = gt_traj[gt_traj[:, 0] < time + self.T + self.dt]

            return gt_traj[:,2:6]
        
        else:
            return None

    def validator(self, trajectory, track_id, time):
        gt_traj = self.data[track_id]
        gt_traj = gt_traj[gt_traj[:, 0] > time - self.dt]
        gt_traj = gt_traj[gt_traj[:, 0] < time + self.T + self.dt]
        
        final_gt_time = gt_traj[-1, 0]
        interp_time = self.time_range + time
        interp_time = interp_time[interp_time <= final_gt_time]
        num_points = len(interp_time)

        if num_points > 0:
            x = np.interp(interp_time, gt_traj[:, 0], gt_traj[:, 2])
            y = np.interp(interp_time, gt_traj[:, 0], gt_traj[:, 3])

            rmse = np.mean(np.sqrt(np.square(trajectory[:num_points,0] - x) + np.square(trajectory[:num_points,1] - y)))
            
            # print("Number Points", num_points)

            # print("GT Traj time", gt_traj[:,0], "real time", self.time_range + time)
            # print(x,y)
            # print("predicted traj x", trajectory[:num_points,0:2])

            # print("RMSE", rmse)
            return rmse
        else:
            return -1


def main():
    rospy.init_node('oncoming_trajectory_prediction', anonymous=True)

    folder = rospy.get_param('oncoming_validation_folder', '/home/karmesh/delta_ws/src/delta_prediction/validation_dataset')
    file_name = rospy.get_param('oncoming_validation_file', 'oncoming_vehicle')
    ground_truth_track = rospy.get_param('ground_truth_track', '/carla/ego_vehicle/tracks/ground_truth')
    fused_track = rospy.get_param('fused_track', '/delta/tracking_fusion/tracker/tracks')

    print('Folder for validation:', folder)
    print('File for validation:', file_name)


    oncoming_validator = OncomingVehicleTrajectoryValidation(folder, 0.1, 2, 2.1)
    rospy.Subscriber(ground_truth_track, TrackArray, oncoming_validator.tracker_gt_callback)

    try:
        while not rospy.is_shutdown():
            rospy.spin()
        oncoming_validator.save(file_name)
        rospy.loginfo('Shutting down')
    except rospy.ROSInterruptException:
        oncoming_validator.save(file_name)
        rospy.loginfo('Shutting down')

if __name__ == "__main__":
    file_name = "oncoming_validation"
    folder = "../validation_dataset"
    main()