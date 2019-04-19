#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Prateek Parmeshwar
Email   : pparmesh@andrew.cmu.edu
Date    : Apr 08, 2019
'''

import carla

from carla_ros_bridge.bridge import CarlaRosBridge
from carla_ros_bridge.bridge_with_rosbag import CarlaRosBridgeWithBag

from nav_msgs.msg import Odometry

import rospy
import tf
import math

import egoTrajPrediction as ETP

def egoTrajPred_callback(odom):
	# Find x and y position
	xPos, yPos = odom.pose.pose.positon.x, odom.pose.pose.positon.y
	# Find x and y velocity components
	xVel, yVel = odom.twist.twist.linear.x, odom.twist.twist.linear.y 
	# Find roll, pitch and yaw
	(roll, pitch, yaw) = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x, 
																   odom.pose.pose.orientation.y, 
																   odom.pose.pose.orientation.z, 
																   odom.pose.pose.orientation.w])

	# Find yaw rate and determine model to use
	yawRate = odom.twist.twist.angular.z

	# If yaw rate is less than 0.01 use the CA model, else use the CTRV model
	if abs(yawRate) < 0.01:
		ego_vehicleState = ETP.EgoTrajPrediction(0.1)
		xAcc, yAcc = 0, 0
		states, covariance = ego_vehicleState.constantAcceleration(xPos, yPos, xVel, yVel, xAcc, yAcc)

	else:
		ego_vehicleState = ETP.EgoTrajPrediction(0.1)
		vel = math.sqrt(xVel**2 + yVel**2)
		states, covariance = ego_vehicleState.constantAcceleration(xPos, yPos, yaw, vel, yawRate)




def egoTrajPred():
	rospy.init_node('egoTrajPred', anonymous = True)

	rospy.Subsciber("/carla/ego_vehicle/odometry", Odometry, egoTrajPred_callback)

	rospy.spin()