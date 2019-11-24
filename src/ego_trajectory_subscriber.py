#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Prateek Parmeshwar
Email   : pparmesh@andrew.cmu.edu
Date    : Apr 17, 2019
'''

import os
import math
import pdb

import rospy
from nav_msgs.msg import Odometry
from diagnostic_msgs.msg import DiagnosticArray
from visualization_msgs.msg import MarkerArray, Marker
from delta_msgs.msg import EgoStateEstimate, EgoStateEstimateArray

from utils import FPSLogger, make_diagnostics_status
from ego_trajectory_prediction import EgoTrajectoryPrediction

fps_logger = FPSLogger('Ego State Prediction')


def publish_diagnostics(pub):
    msg = DiagnosticArray()
    msg.header.stamp = rospy.Time.now()
    msg.status.append(make_diagnostics_status('ego_vehicle', 'prediction', str(fps_logger.fps)))
    pub.publish(msg)

def ego_prediction_shutdown():
    print('\n\033[95m' + '*' * 30 + ' Ego Prediction Shutdown ' + '*' * 30 + '\033[00m\n')


def main():
    rospy.init_node('ego_trajectory_prediction', anonymous=True)
    ego_predictor = EgoTrajectoryPrediction(0.01, 3.0)

    if rospy.has_param('odom_path'):
        odom_path = rospy.get_param('odom_path')
        print('Odometry File:', odom_path)
        ego_predictor.set_odom_path(odom_path)

    rospy.Subscriber("/carla/ego_vehicle/odometry", Odometry, ego_predictor.callback)
    pub_vis = rospy.Publisher('/delta/prediction/ego', Marker, queue_size=10)
    pub_odom = rospy.Publisher('/delta/prediction/ground_truth', Marker, queue_size=10)
    pub_state = rospy.Publisher('/delta/prediction/ego_vehicle/state', EgoStateEstimate, queue_size=10)
    pub_traj = rospy.Publisher('/delta/prediction/ego_vehicle/trajectory', EgoStateEstimateArray, queue_size=10)
    pub_diag = rospy.Publisher('/delta/prediction/ego_vehicle/diagnostics', DiagnosticArray, queue_size=5)

    r = rospy.Rate(10)
    rospy.on_shutdown(ego_prediction_shutdown)
    
    try:
	    while not rospy.is_shutdown():
	        fps_logger.lap()
	        ego_predictor.run(pub_vis, pub_odom, pub_state, pub_traj)
	        fps_logger.tick()

	        publish_diagnostics(pub_diag)
            r.sleep()

    except rospy.ROSInterruptException:
        rospy.loginfo('Shutting down')


if __name__ == '__main__':
    main()
