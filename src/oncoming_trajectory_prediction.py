#!/usr/bin/env python
# -*- coding: utf-8 -*-

#########################################################################
#####                         Delta Autonomy                        #####
#####                   Written By: Karmesh Yadav                   #####
#####                         Date: 09/09/19                        #####
#########################################################################
import copy
import math
import pprint
import pdb
import sys
import time
import traceback

import numpy as np
import matplotlib.pyplot as plt

import rospy
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from diagnostic_msgs.msg import DiagnosticArray
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from delta_msgs.msg import (LaneMarking,
                            LaneMarkingArray,
                            Track,
                            TrackArray,
                            EgoStateEstimate,
                            OncomingVehicleStateEstimate,
                            OncomingVehicleTrajectory,
                            OncomingVehicleTrajectoryArray)

from utils import FPSLogger, make_diagnostics_status
from oncoming_trajectory_validation import OncomingVehicleTrajectoryValidation


cmap = plt.get_cmap('tab20')
# cmap = plt.get_cmap('Set3')
cmap_colors = [cmap(i) for i in range(20) if i % 2 == 0]
EGO_VEHICLE_FRAME = 'ego_vehicle'


def sign(num):
    if num >= 0: return 1
    else: return -1


class LaneMarkerObject:
    def __init__(self, slopes, intercepts):
        assert len(slopes) == 3 and len(intercepts) == 3, \
            "There are supposed to be 3 lanes"

        self.slopes = np.array(slopes)
        self.intercepts = np.array(intercepts)
    
    def find_lane_width(self):
        """
        Finds the lane width of the two lanes
        """
        slope = np.mean(self.slopes)
        dist1 = abs(self.intercepts[0] - self.intercepts[1])/math.sqrt(slope**2 + 1)
        dist2 = abs(self.intercepts[1] - self.intercepts[2])/math.sqrt(slope**2 + 1)
        return dist1, dist2

    def find_closest_lane_marking_distance(self, pos_x, pos_y):
        """
        Find the closest lane to a point
        """
        dist = []
        for i in range(self.slopes.shape[0]):
            dist.append(self.find_point_to_line_distance(self.slopes[i], self.intercepts[i], pos_x, pos_y))
        return np.argmin(dist), dist[np.argmin(dist)]

    def find_left_right_lane_marking(self, point):
        side_line0 = point[1] - self.slopes[0]*point[0] - self.intercepts[0]
        if side_line0 <= 0:
            return -1, 0
        side_line1 = point[1] - self.slopes[1]*point[0] - self.intercepts[1]
        if side_line1 <= 0:
            return 0, 1
        side_line2 = point[1] - self.slopes[2]*point[0] - self.intercepts[2]
        if side_line2 <= 0:
            return 1, 2
        return 2, -1

    def find_point_to_line_distance(self, m, c, x, y):
        return abs(y - m*x - c)/math.sqrt(m**2 + 1)

    def are_points_same_side(self, point1, point2, line_id):
        assert line_id >= 0 and line_id < 3, "Line ID is not within limits"
        side_point1 = point1[1] - self.slopes[line_id]*point1[0] - self.intercepts[line_id]
        side_point2 = point2[1] - self.slopes[line_id]*point2[0] - self.intercepts[line_id]
        if side_point1 == 0 and side_point2 == 0:
            return 1
        side_point1 = sign(side_point1)
        side_point2 = sign(side_point2)
        if side_point1 == side_point2:
            return 1
        return 0

class Prediction:
    def __init__(self, dt, T, track_id, verbose=False):
        self.dt = dt
        self.T = T
        self.time_range = np.arange(0,self.T+0.1,0.1)
        self.verbose = verbose
        self.track_id = track_id
        self.trajectories = []
        self.cost = []
        self.track = None
        self.lanes = None
        self.ego_state = None
        self.fps_logger = FPSLogger('Temp Prediction')
    
    def plot(self, predicted_trajectory, selected_idx):
        for i in range(len(self.trajectories)):
            if i == selected_idx:
                plt.plot(self.trajectories[i][:,0], self.trajectories[i][:,1], c='red', linewidth=5.0)
            elif self.cost[i] >= 1:
                plt.plot(self.trajectories[i][:,0], self.trajectories[i][:,1], c='grey', alpha=1.0/(self.cost[i]+2))
            else:
                plt.plot(self.trajectories[i][:,0], self.trajectories[i][:,1])
        plt.plot(predicted_trajectory[:,0], predicted_trajectory[:,1], '-b', linewidth=5.0)
        x = np.linspace(-1,15,200)
        plt.plot(x, self.lanes.slopes[0]*x+self.lanes.intercepts[0], '-k', linewidth=3.0, label='1')
        plt.plot(x, self.lanes.slopes[1]*x+self.lanes.intercepts[1], '-k', linewidth=3.0, label='2')
        plt.plot(x, self.lanes.slopes[2]*x+self.lanes.intercepts[2], '-k', linewidth=3.0, label='3')

        # plt.pause(0.1)
        plt.show()

    def make_message(self, trajectory):
        msg = OncomingVehicleTrajectory()
        msg.track_id = self.track_id
        for i in range(len(trajectory)):
            state_msg = OncomingVehicleStateEstimate()
            state_msg.x = trajectory[i,0]
            state_msg.y = trajectory[i,1]
            state_msg.vx = trajectory[i,2]
            state_msg.vy = trajectory[i,3]
            msg.state.append(state_msg)
        return msg

    def make_trajectory(self, trajectory, frame_id='/map', marker_id=0,
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
        marker.type = marker.LINE_STRIP
        marker.action = marker.ADD
        for x, y in trajectory:
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.0
            marker.points.append(point)    
        marker.scale.x = 0.3
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 0.3
        marker.lifetime = rospy.Duration(duration)
        return marker

    def generate_trajectories(self, pos_x, pos_y, vel_x, vel_y, lane_width=3.0):
        self.trajectories = []
        min_vel = -1.0
        max_vel = 1.0
        # self.fps_logger.lap()
        lane_width = np.clip(lane_width, 2.0, 4.0)
        for i in np.arange(-lane_width, lane_width, 0.25):
            y = self.solve_trajectory(pos_y, vel_y, 0.0, pos_y+i, self.T)
            for j in np.arange(min_vel, max_vel, 0.5):
                x = self.solve_trajectory(pos_x, vel_x, 0.0, pos_x+(vel_x+j/2)*self.T, self.T, x1_dot=vel_x+j, x1_ddot=0.0)
                # y_pos, y_vel, y_acc

                self.trajectories.append(self.evaluate_polynomial(x,y))
                
                # left_lane, right_lane = self.lanes.find_left_right_lane_marking([0.0, 0.0])
                cost = j*0.01
                
                # if left_lane == -1 or right_lane == -1:
                #     cost = 1
                # else:
                #     left_side = self.lanes.are_points_same_side([pos_x, pos_y], [j+pos_x, i+pos_y], left_lane)
                #     right_side = self.lanes.are_points_same_side([pos_x, pos_y], [j+pos_x, i+pos_y], right_lane)
                #     if left_side == 0 or right_side == 0:
                #         cost = 1

                self.cost.append(cost)
        # self.fps_logger.tick()
        # sys.stdout.write('\r%s ' % self.fps_logger.get_log())
        # sys.stdout.flush()
    
    def solve_trajectory(self, x0, x0_dot, x0_ddot, x1, T, x1_dot=0, x1_ddot=0):
        a0 = x0
        a1 = x0_dot
        a2 = x0_ddot/2.0
        # solve for the rest 3 using Ax = b
        A = np.array([[1,   T,    T**2], \
                      [3, 4*T,  5*T**2], \
                      [3, 6*T, 10*T**2]])

        b = np.array([[(x1 - x0_dot*T - x0_ddot*(T**2)/2 - x0)/T**3], \
                      [(x1_dot - x0_dot - x0_ddot*T)/(T**2)        ], \
                      [(x1_ddot - x0_ddot)/(2*T)                   ]])
        
        x = np.matmul(np.linalg.pinv(A), b)

        coeff = np.array([x[2,0], x[1,0], x[0,0], a2, a1, a0])

        if self.verbose:
            print("==========================Solve Trajectory==========================")
            print("Inputs \n\tx0: {} \n\tx0_dot: {} \n\tx0_ddot: {} \n\tx1: {} \n\tx1_dot: {} \n\tx1_ddot: {}".format(x0, x0_dot, x0_ddot, x1, x1_dot, x1_ddot))
            print("Coeffs \ta5: {} \ta4: {} \ta3: {} \ta2: {} \ta1: {} \ta0: {}".format(x[2,0], x[1,0], x[0,0], a2, a1, a0))

        return coeff

    def evaluate_polynomial(self, long_coeff, lat_coeff):
        trajectory = np.zeros((len(self.time_range),4))

        long_poly = np.poly1d(long_coeff, variable='t')
        trajectory[:, 0] = np.polyval(long_poly, self.time_range)
        trajectory[:, 2] = np.polyval(long_poly.deriv(), self.time_range)

        lat_poly = np.poly1d(lat_coeff, variable='t')
        trajectory[:, 1] = np.polyval(lat_poly, self.time_range)
        trajectory[:, 3] = np.polyval(lat_poly.deriv(), self.time_range)

        if self.verbose:
            print("==========================Evaluate Polynomial==========================")
            print("__________Longitudinal Data__________ \n {}".format(long_poly))
            print("__________Lateral Data__________ \n {}".format(lat_poly))

        return trajectory
    
    def predict_with_motion_model(self, x, y, vx, vy):
        dt = self.dt
        A = np.array([[1, 0, dt,  0],
                      [0, 1,  0, dt],
                      [0, 0,  1,  0],
                      [0, 0,  0,  1]])
        state_old = np.array([x, y, vx, vy])

        sv = 0.1  # Acceleration process noise
        G = np.array([[dt], [dt], [1.0], [1.0]])

        Q = np.matmul(G, G.T)*(sv**2)

        # State space till 2 seconds into the future at intervals of dt
        predicted_trajectory = []
        predicted_trajectory.append(state_old)

        # Calculating the covariance matrix till 2 seconds
        predicted_covariance = []
        P_0 = np.zeros([4, 4])
        P_old = P_0
        for i in range(len(self.time_range)-1):
            # Propogating state
            state_new = np.matmul(A, state_old.T)
            # Check if velocity zero. 
            if sign(vx) != sign(state_new[2]):
                state_new[2] = 0
            predicted_trajectory.append(state_new)
            state_old = state_new
            # Propogating covariance
            P_new = np.matmul(A, np.matmul(P_old, A.T)) + Q
            P_old = P_new
            predicted_covariance.append(P_new)


        predicted_trajectory = np.asarray(predicted_trajectory)
        predicted_covariance = np.asarray(predicted_covariance)

        if self.verbose:
            print("==========================predict_with_motion_model==========================")
            print("__________State__________ \n {}".format(predicted_trajectory[-1]))
            print("__________Covariance__________ \n {}".format(predicted_covariance[-1]))

        return predicted_trajectory, predicted_covariance

    def trajectory_matching(self, predicted_trajectory):
        error = np.array(self.trajectories)[:,:,:2] - predicted_trajectory[:,:2]
        idx = np.argmin(np.square(error).sum(1).mean(1))
        return idx 

    def run(self, track_data, lane_data, ego_state):
        self.lanes = lane_data
        self.track = track_data
        self.ego_state = ego_state

        left_lane, right_lane = self.lanes.find_left_right_lane_marking([self.track[1],self.track[2]])
        left_width, right_width = self.lanes.find_lane_width()

        if left_lane == -1 or right_lane == -1:
            width = 3.0
            self.generate_trajectories(self.track[1], self.track[2], self.track[3], self.track[4], width)
        elif left_lane == 0:
            self.generate_trajectories(self.track[1], self.track[2], self.track[3], self.track[4], left_width)
        else:
            self.generate_trajectories(self.track[1], self.track[2], self.track[3], self.track[4], right_width)
        
        predicted_trajectory, cov = self.predict_with_motion_model(self.track[1], self.track[2], self.track[3], self.track[4])
        idx = self.trajectory_matching(predicted_trajectory)

        # self.plot(predicted_trajectory, idx)

        return self.trajectories[idx], \
               self.make_message(self.trajectories[idx]), \
               self.make_trajectory(self.trajectories[idx][:,:2], frame_id=EGO_VEHICLE_FRAME, \
                                    marker_id=self.track_id, color=cmap_colors[int(self.track_id % 10)])
    

class OncomingTrajectoryPrediction:
    def __init__(self, dt, T, publishers, folder=None, file_name=None, validation_mode=False, verbose=False):
        self.dt = dt
        self.T = T
        self.time_range = np.arange(0,self.T+0.1,0.1)
        self.verbose = verbose
        self.publishers = publishers
        self.validation_mode = validation_mode
        self.timestamp = None
        self.frame_id = None
        self.predictions = {}
        self.ego_state = None
        self.tracks = None
        self.lanes = None
        self.fps_logger = FPSLogger('Oncoming Prediction')

        if self.validation_mode:
            assert file_name != None and folder != None, "Validation Mode is on, please give the file and folder name"
            self.validation = OncomingVehicleTrajectoryValidation(folder, self.dt, self.T)
            self.validation.load(file_name)
            self.validation.filter_data()
            self.RMSE_window = []
    
    def tracker_callback(self, track_msg):
        self.tracks = []
        self.timestamp = track_msg.header.stamp
        self.frame_id = track_msg.header.frame_id
        for i in range(len(track_msg.tracks)):
            self.tracks.append(np.array([track_msg.tracks[i].track_id,
                                         track_msg.tracks[i].x,
                                         track_msg.tracks[i].y,
                                         track_msg.tracks[i].vx,
                                         track_msg.tracks[i].vy]))

    def ego_state_callback(self, ego_state_msg):
        explicit_quat = [ego_state_msg.pose.orientation.x, \
                         ego_state_msg.pose.orientation.y, \
                         ego_state_msg.pose.orientation.z, \
                         ego_state_msg.pose.orientation.w]
        self.ego_state = np.array([[ego_state_msg.pose.position.x,
                                    ego_state_msg.pose.position.y,
                                    euler_from_quaternion(explicit_quat)[2],
                                    ego_state_msg.twist.linear.x,
                                    ego_state_msg.twist.linear.y,
                                    ego_state_msg.twist.angular.z]])


    def lane_marking_callback(self, lane_msg):
        slopes = []
        intercepts = []
        for i in range(len(lane_msg.lanes)):
            slopes.append(lane_msg.lanes[i].slope)
            intercepts.append(lane_msg.lanes[i].intercept)
        self.lanes = LaneMarkerObject(slopes, intercepts)

    def uncompensate_velocity(self, track):
        if self.validation_mode:
            track[3] = track[3] - self.ego_state[0,3]
            track[4] = track[4] - self.ego_state[0,4]
        # print('ego vx', self.ego_state[0,3], 'ego vy', self.ego_state[0,4])
        # print('track vx', track[3], 'track vy', track[4])
        return track

    def run(self):
        traj_array_msg = OncomingVehicleTrajectoryArray()
        traj_array_msg.header.stamp = self.timestamp
        traj_array_msg.header.frame_id = self.frame_id

        vis_array_msg = MarkerArray()
        vis_array_gt_msg = MarkerArray()

        RMSE_sum = 0
        valid_tracks = 0

        self.fps_logger.lap()
        if self.tracks != None and np.all(self.ego_state != None) and self.lanes != None:
            try:
                for i in range(len(self.tracks)):
                    if not self.tracks[i][0] in self.predictions.keys():
                        self.predictions[self.tracks[i][0]] = Prediction(self.dt, self.T, self.tracks[i][0], self.verbose)

                    traj_np, traj_msg, traj_vis = self.predictions[self.tracks[i][0]].run(self.uncompensate_velocity(copy.deepcopy(self.tracks[i])), self.lanes, self.ego_state)
                    traj_array_msg.trajectory.append(traj_msg)
                    vis_array_msg.markers.append(traj_vis)
                    if self.validation_mode:
                        # Visualization
                        trajectory_gt = self.validation.get_gt_trajectory(self.tracks[i][0], self.timestamp.to_sec())

                        if trajectory_gt is not None:
                            traj_vis_gt = self.predictions[self.tracks[i][0]].make_trajectory(trajectory_gt[:,:2], frame_id=EGO_VEHICLE_FRAME, \
                                        marker_id=self.tracks[i][0], color=cmap(int(i)+5))
                            vis_array_gt_msg.markers.append(traj_vis_gt)

                            # Error
                            tmp = self.validation.validator(traj_np, self.tracks[i][0], self.timestamp.to_sec())
                            if tmp != -1:
                                RMSE_sum += tmp 
                                valid_tracks += 1
                        else:
                            print(self.tracks[i][0])

                if self.validation_mode:
                    if valid_tracks != 0: 
                        RMSE_sum /= valid_tracks
                        self.RMSE_window.append(RMSE_sum)

                    if len(self.RMSE_window) > 5:
                        window_err = np.mean(np.asarray(self.RMSE_window[-5:]))
                        sys.stdout.write("\r\033[94m%s Oncoming Vehicle Prediction Error %.3f m %s\033[00m" % ("*"*20, window_err, "*"*20))
                        sys.stdout.flush()
                    self.publishers['traj_vis_gt_pub'].publish(vis_array_gt_msg)

                self.publishers['traj_pub'].publish(traj_array_msg)
                self.publishers['traj_vis_pub'].publish(vis_array_msg)

            except IndexError or KeyError:
                # traceback.print_exc()
                pass

        else:
            time.sleep(1e-5)  # For fps logging

        self.fps_logger.tick()
        # sys.stdout.write('\r%s ' % self.fps_logger.get_log())
        # sys.stdout.flush()
        self.publish_diagnostics()

    def publish_diagnostics(self):
        msg = DiagnosticArray()
        msg.header.stamp = rospy.Time.now()
        msg.status.append(make_diagnostics_status('oncoming_vehicle', 'prediction', str(self.fps_logger.fps)))
        self.publishers['diag_pub'].publish(msg)


def main():
    rospy.init_node('oncoming_trajectory_prediction', anonymous=True)

    # Handle params and topics
    folder = rospy.get_param('oncoming_validation_folder', '/home/karmesh/delta_ws/src/delta_prediction/validation_dataset')
    file_name = rospy.get_param('oncoming_validation_file', 'oncoming_vehicle')
    validation_mode = rospy.get_param('validation_mode', False)

    print('Folder for validation:', folder)
    print('File for validation:', file_name)
    print('Validation Mode:', validation_mode)

    # Publish output topic
    publishers = {}
    publishers['traj_pub'] = rospy.Publisher("/delta/prediction/oncoming_vehicle/trajectory", OncomingVehicleTrajectoryArray, queue_size=5)
    publishers['traj_vis_pub'] = rospy.Publisher("/delta/prediction/oncoming_vehicle/visualization", MarkerArray, queue_size=5)
    publishers['traj_vis_gt_pub'] = rospy.Publisher("/delta/prediction/oncoming_vehicle/visualization_gt", MarkerArray, queue_size=5)
    publishers['diag_pub'] = rospy.Publisher("/delta/prediction/oncoming_vehicle/diagnostics", DiagnosticArray, queue_size=5)

    if validation_mode:
        prediction_time = 2.0
    else:
        prediction_time = 3.0
    oncoming_predictor = OncomingTrajectoryPrediction(0.1, prediction_time, publishers, folder, file_name, validation_mode, False)

    # Subscribe to topics
    if validation_mode:
        rospy.Subscriber('/carla/ego_vehicle/tracks/ground_truth', TrackArray, oncoming_predictor.tracker_callback)
    else:
        rospy.Subscriber('/delta/tracking_fusion/tracker/tracks', TrackArray, oncoming_predictor.tracker_callback)
    rospy.Subscriber('/delta/prediction/ego_vehicle/state', EgoStateEstimate, oncoming_predictor.ego_state_callback)
    rospy.Subscriber('/delta/perception/lane_detection/markings', LaneMarkingArray, oncoming_predictor.lane_marking_callback)

    # In future if we use the OG for penalising invalid  predicted trajectories
    # rospy.Subscriber('/delta/tracking_fusion/tracker/occupancy_grid', OccupancyGrid, oncoming_predictor.tracker_callback)
    
    r = rospy.Rate(10)

    try:
        while not rospy.is_shutdown():
            oncoming_predictor.run()
            r.sleep()
        if validation_mode:
            window_err = np.mean(np.asarray(oncoming_predictor.RMSE_window))
            sys.stdout.write("\r\033[94m%s Average Vehicle Prediction Error %.3f m %s\033[00m" % ("*"*20, window_err, "*"*20))
    except rospy.ROSInterruptException:
        rospy.loginfo('Shutting down')


if __name__ == "__main__":
    main()
