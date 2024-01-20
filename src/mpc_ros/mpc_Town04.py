#! /usr/bin/env python3

import numpy as np
from collections import namedtuple
from math import atan2, sin, cos, pi

import rospy
from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import Twist, PoseStamped, Quaternion
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import MarkerArray

from carla_msgs.msg import CarlaEgoVehicleControl
from mpc_ros.mpc.mpc_path_tracking import mpc_path_tracking
from mpc_ros.utils.curve_generator import curve_generator
from mpc_ros.utils.trans import quat_to_yaw

car = namedtuple('car', 'G g cone_type wheelbase abs_speed abs_steer abs_acce abs_acce_steer')
obstacle = namedtuple('obstacle', 'A b cone_type ')
scaling = 10

def ref_path_list_to_path(ref_path_list):
        path = Path()

        path.header.seq = 0
        path.header.stamp = rospy.get_rostime()
        path.header.frame_id = 'map'

        for i in range(len(ref_path_list)):
            ps = PoseStamped()
            ps.pose.position.x = ref_path_list[i][0, 0]
            ps.pose.position.y = ref_path_list[i][1, 0]
            ps.pose.orientation.w = 1

            path.poses.append(ps)

        return path

def path_to_ref_path_list(path):
        # ref_path_list = [[x, y, yaw], ...]
        ref_path_list = []
        for pose in path.poses:
            p = np.array([pose.pose.position.x, pose.pose.position.y, quat_to_yaw(pose.pose.orientation)])
            ref_path_list.append(p.reshape(3, 1))
        return ref_path_list


class mpc_core:
    def __init__(self):
        
        # ros parameter
        odom_topic= rospy.get_param('odom_topic', '/carla/ego_vehicle/odometry')
        ctl_topic= rospy.get_param('ctl_topic', '/carla/ego_vehicle/vehicle_control_cmd')
        receding = rospy.get_param('receding', 10)
        max_speed = rospy.get_param('max_speed', 8)
        ref_speed = rospy.get_param('ref_speed', 4)
        max_acce = rospy.get_param('max_acce', 1)
        max_acce_steer = rospy.get_param('max_acce_steer', 0.02)
        sample_time = rospy.get_param('sample_time', 0.2)
        max_steer = rospy.get_param('max_steer', 0.5)
        wheelbase = rospy.get_param('wheelbase', 2.87)
        waypoints_topic = rospy.get_param('waypoints_topic', '/carla/ego_vehicle/waypoints')
        self.shape = rospy.get_param('shape', [4.69, 1.85, 2.87, 1.75])   # [length, width, wheelbase, wheelbase_w] limo
     
        cs = rospy.get_param('cs', [1, 1, 1])
        cs = np.diag(cs)
        cu = rospy.get_param('cu', 1)
        cst = rospy.get_param('cst', [1, 1, 1])
        cst = np.diag(cst)
        cut = rospy.get_param('cut', 1)
        
        self.name = 'Town04'
        self.vehicle_list = []
        self.vel_linear = []
        self.vel_steer = []
        self.time_record = []
        self.marker_id = 0
        self.marker_array = MarkerArray()
        self.marker_array_car = MarkerArray()

        # mpc
        self.mpc_track = mpc_path_tracking(receding=receding, max_speed=max_speed, ref_speed=ref_speed, max_acce=max_acce, max_acce_steer=max_acce_steer, sample_time=sample_time, max_steer=max_steer, wheelbase=wheelbase, cs=np.diag([1, 1, 1]), cu=cu, cst=cst, cut=cut)
        rospy.init_node('mpc_node')


        self.vel = Twist()
        # self.output = CarlaEgoVehicleControl()
        self.output = AckermannDrive()
        self.robot_state = np.zeros((3, 1), dtype=np.float32)
        # self.x = self.y = self.z = self.angle = 0
        self.ref_path_list = None
        self.path = None

        self.has_odom = False
        self.has_path = False
        
        # Init subscribers and publishers
        self.sub_waypoints = rospy.Subscriber(waypoints_topic, Path, self.waypoints_callback)
        self.sub_odom = rospy.Subscriber(odom_topic, Odometry, self.robot_state_callback)
        self.pub_marker_car = rospy.Publisher('car_marker', MarkerArray, queue_size=10)
        self.pub_vel = rospy.Publisher(ctl_topic, AckermannDrive, queue_size=10)
        self.pub_path = rospy.Publisher('dubin_path', Path, queue_size=10)
        self.pub_opt_path = rospy.Publisher('opt_path', Path, queue_size=10)

        # Init timer for control loop
        control_frequency = 10
        dur = rospy.Duration(1.0 / control_frequency)
        self.mpc_loop = rospy.Timer(period=dur, callback=self.cal_vel)

    def cal_vel(self, event):

        if self.has_odom == False or self.has_path == False:
            return

        opt_vel, info, is_arrived, _ = self.mpc_track.controller(self.robot_state, 
                                                                 self.ref_path_list, 
                                                                 iter_num=5)

        # rospy.loginfo('states: {}'.format(self.robot_state.reshape(1, 3)))
            
        if is_arrived == True:
            rospy.loginfo('arrived')
            self.vel.linear.x = 0
            self.vel.angular.z = 0
            # self.output.throttle = 0
            # self.output.brake = 0.5
            self.output.speed = 0
            self.output.steering_angle = 0
            self.has_path = False
        else:
            # self.vel.linear.x = round(opt_vel[0, 0], 2)
            # self.vel.angular.z = round(opt_vel[1, 0], 2)
            # self.output.throttle = round(opt_vel[0, 0], 2) / scaling
            # self.output.steer = - round(opt_vel[1, 0], 2)
            self.output.speed = opt_vel[0, 0]
            self.output.steering_angle = - round(opt_vel[1, 0], 2)


        # rospy.loginfo('actions: linear {} steer {}'.format(round(opt_vel[0, 0], 2), round(opt_vel[1, 0], 2)))
            
        self.pub_vel.publish(self.output)
        self.pub_path.publish(self.path)
        self.pub_marker_car.publish(self.marker_array_car)


    def waypoints_callback(self, data: Path):
        if(len(data.poses) == 0):
            return

        rospy.loginfo('get waypoints')

        raw_ref_path_list = path_to_ref_path_list(data)
        cg = curve_generator(point_list=raw_ref_path_list, curve_style='dubins', min_radius=0.5)
        self.ref_path_list = cg.generate_curve(step_size=0.1)
        self.path = ref_path_list_to_path(self.ref_path_list)
        self.pub_path.publish(self.path)
        self.has_path = True


    def robot_state_callback(self, data):
        
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        z = data.pose.pose.position.z

        quat = data.pose.pose.orientation

        yaw = quat_to_yaw(quat)

        yaw_degree = quat_to_yaw(quat)*180/pi

        self.x = x
        self.y = y
        self.z = z
        self.angle = yaw_degree

        offset = self.shape[2] / 2
        
        self.robot_state[0] = x - offset * cos(yaw)
        self.robot_state[1] = y - offset * sin(yaw)
        self.robot_state[2] = yaw
        
        self.has_odom = True


