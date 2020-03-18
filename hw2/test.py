from math import pi, sqrt, atan2
import numpy as np

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose, Twist
from ackermann_msgs.msg import AckermannDriveStamped

from transformations import euler_from_quaternion
from constants import CONSTS
from utils import shortest_angular_distance


def waypoint_cb(msg):
    global waypoints
    for i in range(len(msg.poses)):
        waypoints[i, 0] = msg.poses[i].position.x
        waypoints[i, 1] = msg.poses[i].position.y
        waypoints[i, 2] = euler_from_quaternion([msg.poses[i].orientation.x, msg.poses[i].orientation.y, msg.poses[i].orientation.z, msg.poses[i].orientation.w])[2]


def vehicle_state_cb(msg):
    global rear_axle_center, rear_axle_theta, rear_axle_velocity
    rear_axle_center.position.x = msg.pose.pose.position.x
    rear_axle_center.position.y = msg.pose.pose.position.y
    rear_axle_center.orientation = msg.pose.pose.orientation

    rear_axle_theta = euler_from_quaternion(
    [rear_axle_center.orientation.x, rear_axle_center.orientation.y, rear_axle_center.orientation.z,
        rear_axle_center.orientation.w])[2]

    rear_axle_velocity.linear = msg.twist.twist.linear
    rear_axle_velocity.angular = msg.twist.twist.angular


def pursuitToWaypoint(waypoint):
    global rear_axle_center, rear_axle_theta, rear_axle_velocity, cmd_pub
    rospy.wait_for_message("/ackermann_vehicle/ground_truth/state", Odometry, 5)
    dx = waypoint[0] - rear_axle_center.position.x
    dy = waypoint[1] - rear_axle_center.position.y
    target_distance = sqrt(dx*dx + dy*dy)

    cmd = AckermannDriveStamped()
    cmd.header.stamp = rospy.Time.now()
    cmd.header.frame_id = "base_link"
    cmd.drive.speed = rear_axle_velocity.linear.x
    cmd.drive.acceleration = CONSTS['max_acc']

    while target_distance > CONSTS['waypoint_tol']:
        dx = waypoint[0] - rear_axle_center.position.x
        dy = waypoint[1] - rear_axle_center.position.y
        lookahead_dist = np.sqrt(dx * dx + dy * dy)
        lookahead_theta = atan2(dy, dx)
        alpha = shortest_angular_distance(rear_axle_theta, lookahead_theta)

        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = "base_link"
        # Publishing constant speed of 1m/s
        cmd.drive.speed = 1

        # Reactive steering
        if alpha < 0:
            st_ang = max(-CONSTS['max_steering_angle'], alpha)
        else:
            st_ang = min(CONSTS['max_steering_angle'], alpha)

        cmd.drive.steering_angle = st_ang

        target_distance = sqrt(dx * dx + dy * dy)
        print(cmd)
        cmd_pub.publish(cmd)
        rospy.wait_for_message("/ackermann_vehicle/ground_truth/state", Odometry, 5)


if __name__ == '__main__':
    rospy.init_node('pure_pursuit')
    cmd_pub = rospy.Publisher('/ackermann_vehicle/ackermann_cmd', AckermannDriveStamped, queue_size=10)

    waypoints = np.zeros((CONSTS['num_waypoints'], 3))
    rospy.Subscriber("/ackermann_vehicle/waypoints", PoseArray, waypoint_cb)
    rospy.wait_for_message("/ackermann_vehicle/waypoints", PoseArray, 5)

    rear_axle_center = Pose()
    rear_axle_velocity = Twist()
    rospy.Subscriber("/ackermann_vehicle/ground_truth/state", Odometry, vehicle_state_cb)
    rospy.wait_for_message("/ackermann_vehicle/ground_truth/state", Odometry, 5)

    print(repr(waypoints))

    # for w in waypoints:
        # pursuitToWaypoint(w)
