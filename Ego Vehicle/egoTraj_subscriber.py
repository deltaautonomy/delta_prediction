#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Prateek Parmeshwar
Email   : pparmesh@andrew.cmu.edu
Date    : Apr 17, 2019
'''

import rospy
import tf
import math

import EgoTrajPrediction as ETP

from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray, Marker


def egoTrajPred():
	rospy.init_node('egoTrajPred', anonymous=True)

	ego_vehicleState = ETP.EgoTrajPrediction(0.1, 3.0)

	rospy.Subscriber("/carla/ego_vehicle/odometry", Odometry, ego_vehicleState.ROSCallback)
	pub = rospy.Publisher('delta/prediction/ego', Marker, queue_size=10)

	r = rospy.Rate(10)

	# Randomly publish some data
	while not rospy.is_shutdown():
		ego_vehicleState.run(pub)
		r.sleep()
	#rospy.spin()


if __name__ == '__main__':
	egoTrajPred()
