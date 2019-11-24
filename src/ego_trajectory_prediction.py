''' 
This script predicts the state of the ego-vehicle 3 seconds into the future
Author  : Prateek Parmeshwar
Email   : pparmesh@andrew.cmu.edu
Date    : Apr 16, 2019
'''

import os
import sys
import math
import pdb

import tf
import numpy as np

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from delta_msgs.msg import EgoStateEstimate, EgoStateEstimateArray
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from numpy.linalg import inv
from collections import OrderedDict

cos = lambda theta: np.cos(theta)
sin = lambda theta: np.sin(theta)


class EgoTrajectoryPrediction:
    '''This class instantiates the trajectory prediction for 
    ego vehicle with a filter step --> dt and vehicle ID --> id'''

    def __init__(self, dt, T):
        self.dt = dt
        self.T = T
        self.odom_msg = Odometry()
        self.validation = False
        self.odom = None
        self.RMSE_window = []
        self.old_state = np.zeros(5)
        self.ego_validation_array = []
        self.validation_dict = OrderedDict()
        self.validation_odom = []
        self.current_transform = np.zeros([3,3])

    # We will make the following forward proprogation models here.
    # 1. Constant acceleration model
    # 2. Constant Turn Rate and Speed

    # ------------ Constant Acceleration-------------------------

    # The constant acceleration model assumes longitudinal motion

    def set_odom_path(self, odom_path):
        if self.odom is None:
            self.validation = True
            self.odom = np.load(odom_path)

    def constant_acceleration(self, xPos, yPos, xVel, yVel, xAcc, yAcc):
        ''' Future state is x(k+1)
        x(k+1) = A(k)*x(K) + w where w is 0 mean Gaussian process noise
        This process noise is 0 always'''

        # Make the transition matrix
        dt = self.dt
        T = self.T

        A = np.array(
            [
                [1.0, 0.0, dt, 0.0, 0.5*(dt**2), 0.0],
                [0.0, 1.0, 0.0, dt, 0.0, 0.5*(dt**2)],
                [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        # Make the state space matrix
        x_0 = np.array([xPos, yPos, xVel, yVel, xAcc, yAcc])
        t = 0  # initialize time

        # Evaluate reliability of estimated path, propogate covariance
        # P(k+1) = A(k)*P(k) + Q where Q is the covariance matrix of process noise
        # Q = G*G'*sa**2. G and sa have been defined below

        sa = 0.1  # Acceleration process noise
        G = np.array([[0.5*(dt**2)], [0.5*(dt**2)], [dt], [dt], [1.0], [1.0]])

        Q = np.matmul(G, G.T)*(sa**2)

        # State space till 3 seconds into the future at intervals of dt
        stateSpacePrediction = []
        x_old = x_0

        # Calculating the covariance matrix till 3 seconds
        covarianceTotal = []
        P_0 = np.zeros([6, 6])
        P_old = P_0

        while t <= T:
            # Propogating state
            x_new = np.matmul(A, x_old.T)
            # Check if velocity zero. 
            if self.sign(xVel) != self.sign(x_new[2]):
                x_new[2] = 0
            stateSpacePrediction.append(x_new)
            x_old = x_new
            # Propogating covariance
            P_new = np.matmul(A, np.matmul(P_old, A.T)) + Q
            P_old = P_new
            covarianceTotal.append(P_new)
            # Increment time
            t = t + dt

        stateSpacePrediction = np.asarray(stateSpacePrediction)
        covarianceTotal = np.asarray(covarianceTotal)

        return stateSpacePrediction, covarianceTotal

    def sign(self, num):
        if num >= 0: return 1
        else: return -1

    # ------------ Constant Turn Rate and Speed (CTR)-------------------------

    def constant_turn_rate(self, xPos, yPos, yaw, vel, yaw_rate):
        ''' This model assumes a constant turn rate, i.e, yaw rate and a constant
        velocity which is perpendicular to its acceleration'''

        dt = self.dt  # Kalman-Filter time-step
        T = self.T

        # Evaluate reliability of estimated path, propogate covariance
        # P(k+1) = A(k)*P(k) + Q where Q is the covariance matrix of process noise

        sGPS = 0.5 * 8.8 * (dt ** 2)  # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
        sCourse = 0.1 * dt  # assume 0.1rad/s as maximum turn rate for the vehicle
        sVelocity = 8.8 * dt  # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
        sYaw = 1.0 * dt  # assume 1.0rad/s2 as the maximum turn rate acceleration for the vehicle

        Q = np.diag([sGPS ** 2, sGPS ** 2, sCourse ** 2, sVelocity ** 2, sYaw ** 2])
        
        # Initialize state and P matrix
        x_0 = np.array([xPos, yPos, yaw, vel, yaw_rate])
        P_0 = np.zeros([5, 5])

        # Predict trajectory for 3 seconds into the future
        x_old = x_0
        P_old = P_0
        stateSpacePrediction = []
        covarianceTotal = []

        t = 0  # initialize time

        while t <= T:
            # Making the transistion matrix (linearizing using Jacobian)
            # Assign values to state variables
            xPos, yPos, yaw, vel, yaw_rate = x_old

            A13 = (vel / yaw_rate) * (-cos(yaw) + cos(dt * yaw_rate + yaw))
            A14 = (1 / yaw_rate) * (-sin(yaw) + sin(dt * yaw_rate + yaw))
            A15 = dt * (vel / yaw_rate) * cos(dt * yaw_rate + yaw) - (vel / (yaw_rate ** 2)) * (
                -sin(yaw) + sin(dt * yaw_rate + yaw))
            A23 = (vel / yaw_rate) * (-sin(yaw) + sin(dt * yaw_rate + yaw))
            A24 = (1 / yaw_rate) * (cos(yaw) - cos(dt * yaw_rate + yaw))
            A25 = dt * (vel / yaw_rate) * sin(dt * yaw_rate + yaw) - (vel / (yaw_rate ** 2)) * (
                cos(yaw) - cos(dt * yaw_rate + yaw))

            A = np.array(
                [
                    [1, 0, A13, A14, A15],
                    [0, 1, A23, A24, A25],
                    [0, 0, 1, 0, dt],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                ]
            )

            # Forward propogate state
            x_new = np.matmul(A, x_old)

            stateSpacePrediction.append(x_new)
            x_old = x_new

            # Propogate covariance
            P_new = np.matmul(A, np.matmul(P_old, A.T)) + Q
            covarianceTotal.append(P_new)
            P_old = P_new
            t = t + dt

        stateSpacePrediction = np.asarray(stateSpacePrediction)
        covarianceTotal = np.asarray(covarianceTotal)

        return stateSpacePrediction, covarianceTotal

    def visualize(self, states, colour, pub):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.stamp = self.odom_msg.header.stamp
        marker.header.frame_id = 'ego_vehicle'
        marker.ns = 'predicted_trajectory'
        marker.type = 4
        marker.action = 0 # Adds an object (check it later)
        for idx in range(states.shape[0]):
            # state in both models have x and y first   
            P = Point()
            P.x = states[idx, 0] # state in both models have x and y first
            P.y = states[idx, 1]
            P.z = 0
            marker.points.append(P)
        marker.scale.x = 2
        marker.color.a = colour[0]
        marker.color.r = colour[1]
        marker.color.g = colour[2]
        marker.color.b = colour[3]
        # marker.frame_locked = True
        # marker_array.markers.append(marker)
        pub.publish(marker)
        
    def ego_vehicle_transform(self, x, y, yaw):
        ego_world = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), x],
                [np.sin(yaw), np.cos(yaw), y],
                [0, 0, 1],
            ]
        )
        return ego_world

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def prediction_validator(self, states, current_time, pub_odom, ego_world_transform):
        # Find total number of predictions 
        num_predictions = int(self.T / self.dt)
        colour_odom = [1.0, 0.0, 0.0, 1.0 ]
        # pdb.set_trace()
        try:
            # Find odometry values to be compared. They are only a portion of the entire data seq
            odometry_states = []
            odom_timestep = 0
            while odom_timestep <= self.T:
                current_time += self.dt
                # index = self.odom[:, 2].searchsorted(current_time)
                index = self.find_nearest(self.odom[:,2],current_time)
                odom_timestep += self.dt
                val = self.odom[index, 0:2]
                odometry_states.append(val)

            odometry_states = np.asarray(odometry_states)
            self.visualize(odometry_states, colour_odom, pub_odom)

            predicted_states = states[:,0:2]
            r,c = np.shape(predicted_states)
            predicted_states = np.c_[predicted_states,np.ones(r)]
            # Transform predicted states to world frame
            predicted_states = np.matmul(ego_world_transform,predicted_states.T)
            predicted_states = predicted_states.T
            predicted_states = predicted_states[:,0:2]
            # print(predicted_states)

            # Find the RMSE error between the predicted states and odometry
            err =  np.linalg.norm((predicted_states - odometry_states), axis = 1)
            
            # Evaluate RMSE
            RMSE = np.sum(err) / num_predictions
            self.RMSE_window.append(RMSE)

            # Find mean 
            if len(self.RMSE_window) > 100:
                window_err = self.RMSE_window[-100:]
                window_err = np.asarray(window_err)
                window_err = np.mean(window_err)
                self.ego_validation_array.append(window_err)
                sys.stdout.write("\r\033[94m%s Prediction Error %.3f m %s\033[00m" % ("*"*20, window_err, "*"*20))
                sys.stdout.flush()

        except Exception as e:
            print("Exception")
            pass
    def publish_validation(self):
        for err in self.ego_validation_array:
            print("Ego Prediction Error: %.3f\n"%(err))
            # sys.stdout.write("\r\033[94m%s Prediction Error %.3f m %s\033[00m" % ("*"*20, err, "*"*20))
            # sys.stdout.flush()

    def find_acceleration(self, xPos, yPos, xVel, yVel, curr_time):
        if self.old_state[0] != 0 and self.old_state[1] != 0:
            if (xPos != self.old_state[0]):
                # xAcc = abs((xVel**2 - self.old_state[2]**2) / (2*(xPos - self.old_state[0])))
                xAcc = (xVel - self.old_state[2])/(curr_time-self.old_state[4])
            else:
                xAcc = 0
            
            if (yPos != self.old_state[1]):
                # yAcc = (yVel**2 - self.old_state[3]**2) / (2*(yPos - self.old_state[1]))
                yAcc = (yVel - self.old_state[3])/(curr_time-self.old_state[4])
            else:
                yAcc = 0
        else:
            xAcc = 0
            yAcc = 0
        
        self.old_state = np.array([xPos, yPos, xVel, yVel, curr_time])
        return xAcc, yAcc

    def publish_state(self, motion_model, states, pub, accel):
        '''Helper function to publish ego-vehicle state for current timestep'''
        ego_state = EgoStateEstimate()
        ego_state.header.stamp = self.odom_msg.header.stamp
        ego_state.pose = self.odom_msg.pose.pose
        ego_state.twist = self.odom_msg.twist.twist
        ego_state.accel.linear.x = accel[0]
        ego_state.accel.linear.y = accel[1]
        pub.publish(ego_state)

    def publish_trajectory(self, motion_model, states, pub):
        '''Helper function to publish ego-vehicle trajectory for the next n time steps'''
        ego_state_array = EgoStateEstimateArray()
        states_list = list(states)
        # Iterate over states
        for state in states_list:
            # Choose motion model
            # CA state: [xPos, yPos, xVel, yVel, xAcc, yAcc]
            if motion_model == 'CA':
                ego_state = EgoStateEstimate()
                ego_state.header.stamp = self.odom_msg.header.stamp
                
                # Position
                ego_state.pose.position.x = state[0]
                ego_state.pose.position.y = state[1]
                
                # Velocity
                ego_state.twist.linear.x = state[2]
                ego_state.twist.linear.y = state[3]
                
                # Acceleration
                ego_state.accel.linear.x = state[4]
                ego_state.accel.linear.y = state[5]
                
                # Append to array
                ego_state_array.states.append(ego_state)

            # CTRV state: [xPos, yPos, yaw, vel, yaw_rate]
            if motion_model == 'CTRV':
                ego_state = EgoStateEstimate()
                ego_state.header.stamp = self.odom_msg.header.stamp
                
                # Position
                ego_state.pose.position.x = state[0]
                ego_state.pose.position.y = state[1]

                # Orientation
                quat = quaternion_from_euler(0.0, 0.0, state[2])
                ego_state.pose.orientation.x = quat[0]
                ego_state.pose.orientation.y = quat[1]
                ego_state.pose.orientation.z = quat[2]
                ego_state.pose.orientation.w = quat[3]

                # Velocity
                ego_state.twist.linear.x = state[3]
                ego_state.twist.angular.z = state[4]
                ego_state_array.states.append(ego_state)

        pub.publish(ego_state_array)

    def callback(self, odom):
        if self.odom is None and self.validation: return
        self.odom_msg = odom

    # def online_validation(self):
    #     num_predictions = int(self.T / self.dt)
    #     try:
    #         debug_count = 0
    #         self.validation_odom = np.asarray(self.validation_odom)
    #         print(self.validation_odom)
    #         print('\n\n\n\n\n\n\n')
    #         for key in self.validation_dict:
    #             current_time = key
    #             # print(current_time)
    #             odometry_states = []
    #             odom_timestep = 0
    #             while odom_timestep <= self.T:
    #                 current_time += self.dt
    #                 # print(type(current_time))
    #                 # index = self.validation_odom[:, 2].searchsorted(current_time)
    #                 index = self.find_nearest(self.validation_odom[:,2],current_time)
    #                 # print(index)
    #                 odom_timestep += self.dt
    #                 val = self.validation_odom[index, 0:2]
    #                 odometry_states.append(val)

    #             odometry_states = np.asarray(odometry_states)
    #             # print(odometry_states)
    #         # self.visualize(odometry_states, colour_odom, pub_odom)

    #             predicted_states = self.validation_dict[key]
    #             r,c = np.shape(predicted_states)
    #             predicted_states = np.c_[predicted_states,np.ones(r)]
    #             # Transform predicted states to world frame
    #             predicted_states = np.matmul(self.current_transform,predicted_states.T)
    #             predicted_states = predicted_states.T
    #             predicted_states = predicted_states[:,0:2]
    #             print(predicted_states)
            

    #             # Find the RMSE error between the predicted states and odometry
    #             err =  np.linalg.norm((predicted_states - odometry_states), axis = 1)
            
    #             # Evaluate RMSE
    #             RMSE = np.sum(err) / num_predictions
    #             self.RMSE_window.append(RMSE)

    #             if len(self.RMSE_window) > 100:
    #                 window_err = self.RMSE_window[-100:]
    #                 window_err = np.asarray(window_err)
    #                 window_err = np.mean(window_err)
    #                 self.ego_validation_array.append(window_err)
    #             debug_count +=1
    #             if (debug_count == 3):
    #                 break

    #     except Exception as e:
    #         print("Exception")
    #         pass
                
    def run(self, pub_pred, pub_odom, pub_state, pub_traj):
        odom_msg = self.odom_msg
        # Motion model variable
        motion_model = ''

        # Find current time
        current_time = odom_msg.header.stamp.to_sec()
        # Find x and y position
        xPos, yPos = odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y
        # print(xPos,yPos,current_time)
        # Find x and y velocity components
        xVel, yVel = odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y
        
        # Find roll, pitch and yaw
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
            [
                odom_msg.pose.pose.orientation.x,
                odom_msg.pose.pose.orientation.y,
                odom_msg.pose.pose.orientation.z,
                odom_msg.pose.pose.orientation.w,
            ]
        )

        # Find yaw rate and determine model to use
        yaw_rate = odom_msg.twist.twist.angular.z
        # xAcc, yAcc = 0, 0
        xAcc, yAcc = self.find_acceleration(xPos, yPos, xVel, yVel, current_time)
        # If yaw rate is less than 0.01 use the CA model, else use the CTRV model
        if np.abs(yaw_rate) < 1e-6:
            motion_model = 'CA'
            states, covariance = self.constant_acceleration(0, 0, xVel, yVel, xAcc, yAcc)
        else:
            motion_model = 'CTRV'
            vel = self.sign(xVel) * math.sqrt(xVel**2 + yVel**2)
            states, covariance = self.constant_turn_rate(0, 0, 0, vel, yaw_rate)


        ego_world_transform = self.ego_vehicle_transform(xPos,yPos,yaw)
        self.current_transform = ego_world_transform
        if self.validation:
            self.prediction_validator(states, current_time, pub_odom, ego_world_transform)
        
        # Online Validation
        # self.validation_dict[current_time] = states[:,0:2]
        # self.validation_odom.append([xPos,yPos,current_time])

        colour_pred = [0.5, 0.30196078431372547, 0.6862745098039216, 0.2901960784313726]
        self.visualize(states, colour_pred, pub_pred)

        # Add publisher for ego state data
        self.publish_state(motion_model, self.odom_msg, pub_state, [xAcc, yAcc])
        self.publish_trajectory(motion_model, states, pub_traj)
