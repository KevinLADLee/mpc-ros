# !/usr/bin/env python3

from geometry_msgs.msg import Quaternion
from math import sin, cos, atan2

def quat_to_yaw(quater):
         
        w = quater.w
        x = quater.x
        y = quater.y
        z = quater.z

        raw = atan2(2* ( w*z + x *y), 1 - 2 * (pow(z, 2) + pow(y, 2)))

        return raw

def yaw_to_quat(yaw):
         
        w = cos(yaw/2)
        x = 0
        y = 0
        z = sin(yaw/2)

        quat = Quaternion(w=w, x=x, y=y, z=z)

        return quat
