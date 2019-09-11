#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Prateek Parmeshwar
Email   : pparmesh@andrew.cmu.edu
Date    : Apr 17, 2019
'''

import os
import math

import rospy
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray, Marker
from carla_ros_bridge.msg import EgoState, EgoStateArray

from ego_trajectory_prediction import EgoTrajectoryPrediction


def main():
	rospy.init_node('ego_trajectory_prediction', anonymous=True)
	ego_predictor = EgoTrajectoryPrediction(0.01, 3.0)
	
	if rospy.has_param('odom_path'):
		odom_path = rospy.get_param('odom_path')
		print('Odometry File:', odom_path)
		ego_predictor.set_odom_path(odom_path)

	rospy.Subscriber("/carla/ego_vehicle/odometry", Odometry, ego_predictor.callback)
	pub_pred = rospy.Publisher('/delta/prediction/ego', Marker, queue_size=10)
	pub_odom = rospy.Publisher('/delta/prediction/ground_truth', Marker, queue_size=10)
	pub_state = rospy.Publisher('/delta/prediction/ego_states', EgoStateArray, queue_size=10)
	r = rospy.Rate(10)

	# Randomly publish some data
	while not rospy.is_shutdown():
		ego_predictor.run(pub_pred, pub_odom, pub_state)
		r.sleep()


if __name__ == '__main__':
	main()
