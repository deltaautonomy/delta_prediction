#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.0.0
Date    : Apr 07, 2019
'''

import time

import rospy
from diagnostic_msgs.msg import DiagnosticStatus


class FPSLogger:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.fps = None
        self.last = None
        self.total_time = 0
        self.total_frames = 0

    def lap(self):
        self.last = time.time()

    def tick(self, count=1):
        self.total_time = time.time() - self.last
        self.total_frames = count
        self.fps = self.total_frames / self.total_time

    def log(self, tick=False):
        if tick: self.tick()
        print('\033[94m %s FPS:\033[00m \033[93m%.1f\033[00m' % (self.name, self.fps))

    def get_log(self, tick=False):
        if tick: self.tick()
        return '\033[94m %s FPS:\033[00m \033[93m%.1f\033[00m' % (self.name, self.fps)


def make_diagnostics_status(name, pipeline, fps, level=DiagnosticStatus.OK):
    msg = DiagnosticStatus()
    msg.level = DiagnosticStatus.OK
    msg.name = name
    msg.message = fps
    msg.hardware_id = pipeline
    return msg
