''' 
This script predicts the state of the ego-vehicle 3 seconds into the future
Author  : Prateek Parmeshwar
Email   : pparmesh@andrew.cmu.edu
Date    : Apr 16, 2019
'''

import numpy as np
import math

cos = lambda theta: math.cos(theta)
sin = lambda theta: math.sin(theta)


class EgoTrajPrediction:
    '''This class instantiates the trajectory prediction for 
	ego vehicle with a filter step --> T and vehicle ID --> id'''

    def __init__(self, T):
        self.T = T

    # We will make the following forward proprogation models here.
    # 1. Constant acceleration model
    # 2. Constant Turn Rate and Speed

    # ------------ Constant Acceleration-------------------------

    # The constant acceleration model assumes longitudinal motion

    def constantAcceleration(self, xPos, yPos, xVel, yVel, xAcc, yAcc):
        ''' Future state is x(k+1)
		x(k+1) = A(k)*x(K) + w where w is 0 mean Gaussian process noise
		This process noise is 0 always'''

        # Make the transition matrix
        T = self.T

        A = np.matrix(
            [
                [1.0, 0.0, T, 0.0, 1 / 2.0 * T ** 2, 0.0],
                [0.0, 1.0, 0.0, T, 0.0, 1 / 2.0 * T ** 2],
                [0.0, 0.0, 1.0, 0.0, T, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, T],
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

        sa = 0.001  # Acceleration process noise
        G = np.matrix([[1 / 2.0 * T ** 2], [1 / 2.0 * T ** 2], [T], [T], [1.0], [1.0]])

        Q = G * G.T * sa ** 2

        # State space till 3 seconds into the future at intervals of T
        stateSpacePrediction = []
        x_old = x_0

        # Calculating the covariance matrix till 3 seconds
        covarianceTotal = []
        P_0 = np.zeros([6, 6])
        P_old = P_0

        while t <= 30:
            # Propogating state
            x_new = np.matmul(A, x_old.T)
            stateSpacePrediction.append(x_new)
            x_old = x_new
            # Propogating covariance
            P_new = np.matmul(A, np.matmul(P_old, A.T)) + Q
            P_old = P_new
            covarianceTotal.append(P_new)
            # Increment time
            t = t + 0.1

        return stateSpacePrediction, covarianceTotal

    # ------------ Constant Turn Rate and Speed (CTR)-------------------------

    def constantTurnRate(self, xPos, yPos, yaw, vel, yawRate):
        ''' This model assumes a constant turn rate, i.e, yaw rate and a constant
		velocity which is perpendicular to its acceleration'''

        T = self.T  # Kalman-Filter time-step

        # Evaluate reliability of estimated path, propogate covariance
        # P(k+1) = A(k)*P(k) + Q where Q is the covariance matrix of process noise

        sGPS = (
            0.5 * 8.8 * T ** 2
        )  # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
        sCourse = 0.1 * T  # assume 0.1rad/s as maximum turn rate for the vehicle
        sVelocity = (
            8.8 * T
        )  # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
        sYaw = (
            1.0 * T
        )  # assume 1.0rad/s2 as the maximum turn rate acceleration for the vehicle

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

        while t <= 30:
            # Making the transistion matrix (linearizing using Jacobian)

            # Assign values to state variables
            xPos, yPos, yaw, vel, yawRate = x_old

            A13 = vel / yawRate * (-cos(yaw) + cos(T * yawRate + yaw))
            A14 = 1 / yawRate * (-sin(yaw) + sin(T * yawRate + yaw))
            A15 = T * vel / yawRate * cos(T * yawRate + yaw) - vel / (yawRate ** 2) * (
                -sin(yaw) + sin(T * yawRate + yaw)
            )
            A23 = vel / yawRate * (-sin(yaw) + sin(T * yawRate + yaw))
            A24 = 1 / yawRate * (cos(yaw) - cos(T * yawRate + yaw))
            A25 = T * vel / yawRate * sin(T * yawRate + yaw) - vel / (yawRate ** 2) * (
                cos(yaw) - cos(T * yawRate + yaw)
            )

            A = np.array(
                [
                    [1, 0, A13, A14, A15],
                    [0, 1, A23, A24, A25],
                    [0, 0, 1, 0, T],
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

            t = t + 0.1

        return stateSpacePrediction, covarianceTotal
