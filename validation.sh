#!/bin/bash

roslaunch delta_prediction ego_vehicle_prediction_validation.launch version:=v1
rosnode kill /egoTraj_subscriber
roslaunch delta_prediction ego_vehicle_prediction_validation.launch version:=v2
rosnode kill /egoTraj_subscriber
roslaunch delta_prediction ego_vehicle_prediction_validation.launch version:=v3
rosnode kill /egoTraj_subscriber
roslaunch delta_prediction ego_vehicle_prediction_validation.launch version:=v4
rosnode kill /egoTraj_subscriber
