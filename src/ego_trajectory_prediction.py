''' 
This script predicts the state of the ego-vehicle 3 seconds into the future
Author  : Prateek Parmeshwar
Email   : pparmesh@andrew.cmu.edu
Date    : Apr 16, 2019
'''

import os
import sys
import math

import tf
import numpy as np

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from carla_ros_bridge.msg import EgoState, EgoStateArray

cos = lambda theta: np.cos(theta)
sin = lambda theta: np.sin(theta)

# Declare globals
ODOM = None
VALIDATION = False
RMSE_window = []
X_OLD = 0
XVEL_OLD = 0
TIME_OLD = 0


class EgoTrajectoryPrediction:
	'''This class instantiates the trajectory prediction for 
	ego vehicle with a filter step --> dt and vehicle ID --> id'''

	def __init__(self, dt, T):
		self.dt = dt
		self.T = T
		self.last_odom_msg = Odometry()

	# We will make the following forward proprogation models here.
	# 1. Constant acceleration model
	# 2. Constant Turn Rate and Speed

	# ------------ Constant Acceleration-------------------------

	# The constant acceleration model assumes longitudinal motion

	@staticmethod
	def set_odom_path(odom_path):
		global ODOM, VALIDATION
		if ODOM is None:
			VALIDATION = True
			ODOM = np.load(odom_path)

	def constant_acceleration(self, xPos, yPos, xVel, yVel, xAcc, yAcc):
		''' Future state is x(k+1)
		x(k+1) = A(k)*x(K) + w where w is 0 mean Gaussian process noise
		This process noise is 0 always'''

		# Make the transition matrix
		dt = self.dt
		T = self.T

		A = np.array(
			[
				[1.0, 0.0, dt, 0.0, 1 / 2.0 * dt ** 2, 0.0],
				[0.0, 1.0, 0.0, dt, 0.0, 1 / 2.0 * dt ** 2],
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
		G = np.array([[1 / 2.0 * dt ** 2], [1 / 2.0 * dt ** 2], [dt], [dt], [1.0], [1.0]])

		Q = np.matmul(G, G.T) * (sa ** 2)

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
			# Check if decelerating. 
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

	def constant_turn_rate(self, xPos, yPos, yaw, vel, yawRate):
		''' This model assumes a constant turn rate, i.e, yaw rate and a constant
		velocity which is perpendicular to its acceleration'''

		dt = self.dt  # Kalman-Filter time-step
		T = self.T

		# Evaluate reliability of estimated path, propogate covariance
		# P(k+1) = A(k)*P(k) + Q where Q is the covariance matrix of process noise

		sGPS = 0.5 * 8.8 * dt ** 2  # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
		sCourse = 0.1 * dt  # assume 0.1rad/s as maximum turn rate for the vehicle
		sVelocity = 8.8 * dt  # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
		sYaw = 1.0 * dt  # assume 1.0rad/s2 as the maximum turn rate acceleration for the vehicle

		Q = np.diag([sGPS ** 2, sGPS ** 2, sCourse ** 2, sVelocity ** 2, sYaw ** 2])
		
		# Initialize state and P matrix
		x_0 = np.array([xPos, yPos, yaw, vel, yawRate])
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
			xPos, yPos, yaw, vel, yawRate = x_old

			A13 = vel / yawRate * (-cos(yaw) + cos(dt * yawRate + yaw))
			A14 = 1 / yawRate * (-sin(yaw) + sin(dt * yawRate + yaw))
			A15 = dt * vel / yawRate * cos(dt * yawRate + yaw) - vel / (yawRate ** 2) * (
				-sin(yaw) + sin(dt * yawRate + yaw))
			A23 = vel / yawRate * (-sin(yaw) + sin(dt * yawRate + yaw))
			A24 = 1 / yawRate * (cos(yaw) - cos(dt * yawRate + yaw))
			A25 = dt * vel / yawRate * sin(dt * yawRate + yaw) - vel / (yawRate ** 2) * (
				cos(yaw) - cos(dt * yawRate + yaw))

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

	def callback(self, odom):
		if ODOM is None and VALIDATION: return
		self.last_odom_msg = odom

	def run(self, pub_pred, pub_odom, pub_state):
		global TIME_OLD

		last_odom_msg = self.last_odom_msg
		# Motion model variable
		motion_model = ''

		# Find current time
		current_time = last_odom_msg.header.stamp.to_sec()

		# Find x and y position
		xPos, yPos = last_odom_msg.pose.pose.position.x, last_odom_msg.pose.pose.position.y

		# Find x and y velocity components
		xVel, yVel = last_odom_msg.twist.twist.linear.x, last_odom_msg.twist.twist.linear.y
		
		# Find roll, pitch and yaw
		(roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
			[
				last_odom_msg.pose.pose.orientation.x,
				last_odom_msg.pose.pose.orientation.y,
				last_odom_msg.pose.pose.orientation.z,
				last_odom_msg.pose.pose.orientation.w,
			]
		)

		# Find yaw rate and determine model to use
		yawRate = last_odom_msg.twist.twist.angular.z

		# If yaw rate is less than 0.01 use the CA model, else use the CTRV model
		if abs(yawRate) < 0.01:
			motion_model = 'CA'
			xAcc = self.find_acceleration(xPos, xVel)
			yAcc = self.find_acceleration(yPos, yVel)
			states, covariance = self.constant_acceleration(0, 0, xVel, yVel, xAcc, yAcc)
		else:
			motion_model = 'CTRV'
			vel = self.sign(xVel) * math.sqrt(xVel**2 + yVel**2)
			states, covariance = self.constant_turn_rate(0, 0, 0, vel, yawRate)

		if VALIDATION:
			self.prediction_validator(states, current_time, pub_odom)

		colour_pred = [1.0, 0.0, 1.0, 0.0 ]
		self.visualize(states, colour_pred, pub_pred)

		# Add publisher for ego state data
		self.ego_state_publisher(motion_model, states, pub_state)

		TIME_OLD = current_time

	def visualize(self, states, colour, pub):
		marker_array = MarkerArray()
		marker = Marker()
		marker.header.stamp = self.last_odom_msg.header.stamp
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
			
		marker.scale.x = 0.3
		marker.color.a = colour[0]
		marker.color.r = colour[1]
		marker.color.g = colour[2]
		marker.color.b = colour[3]
		# marker.frame_locked = True
		# marker_array.markers.append(marker)
		pub.publish(marker)

	def prediction_validator(self, states, current_time, pub_odom):
		global VALIDATOR_COUNTER, ODOM, RMSE_window
		# Find total number of predictions 
		num_predictions = int(self.T / self.dt)
		colour_odom = [1.0, 0.0, 0.0, 1.0 ]

		try:
			# Find odometry values to be compared. They are only a portion of the entire data seq
			odometry_states = []
			odom_timestep = 0
			while odom_timestep <= self.T:
				current_time += self.dt
				index = ODOM[:,2].searchsorted(current_time)

				odom_timestep += self.dt
				val = ODOM[index,0:2]
				odometry_states.append(val)

			odometry_states = np.asarray(odometry_states)
			self.visualize(odometry_states, colour_odom, pub_odom)

			predicted_states = states[:,0:2]

			# Find the RMSE error between the predicted states and odometry
			err =  np.linalg.norm((predicted_states - odometry_states), axis = 1)
			
			# Evaluate RMSE
			RMSE = np.sum(err) / num_predictions
			RMSE_window.append(RMSE)

			# Find mean 
			if len(RMSE_window) > 100:
				window_err = RMSE_window[-100:]
				window_err = np.asarray(window_err)
				window_err = np.mean(window_err)
				sys.stdout.write("\r\033[94m%s Prediction Error %.3f m %s\033[00m" % ("*"*20, window_err, "*"*20))
				sys.stdout.flush()

		except Exception as e:
			pass

	def find_acceleration(self, xPos, xVel):
		global X_OLD, XVEL_OLD

		if (xPos!=X_OLD):
			xAcc = (xVel**2 - XVEL_OLD**2) / (2*(xPos - X_OLD))
			X_OLD = xPos
			XVEL_OLD = xVel
			return xAcc
		else:
			X_OLD = xPos
			XVEL_OLD = xVel
			return 0

	def ego_state_publisher(self, motion_model, states, pub):
		'''Helper function to publish ego-vehicle state for the next n time steps'''
		egoState_array = EgoStateArray()
		states_list = list(states)
		# Iterate over states
		for state in states_list:
			# Choose motion model
			if motion_model == 'CA':
				egoState = EgoState()
				egoState.header.stamp = self.last_odom_msg.header.stamp
				egoState.pose.position.x = state[0]
				egoState.pose.position.y = state[1]
				egoState.twist.linear.x = state[2]
				egoState.twist.linear.y = state[3]
				egoState.accel.linear.x = state[4]
				egoState.accel.linear.y = state[5]
				# Append to array
				egoState_array.egoStates.append(egoState)

			if motion_model == 'CTRV':
				egoState = EgoState()
				egoState.header.stamp = self.last_odom_msg.header.stamp
				egoState.pose.position.x = state[0]
				egoState.pose.position.y = state[1]
				egoState.pose.orientation.z = state[2]
				egoState.twist.linear.x = state[3]
				egoState.twist.angular.z = state[4]
				# Append to array
				egoState_array.egoStates.append(egoState)

		pub.publish(egoState_array)
