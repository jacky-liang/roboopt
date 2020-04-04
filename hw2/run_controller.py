import argparse
import os
from math import pi, sqrt, atan2
import numpy as np

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose, Twist
from ackermann_msgs.msg import AckermannDriveStamped

from transformations import euler_from_quaternion
from constants import CS
from utils import shortest_angular_distance

from tensorboardX import SummaryWriter
from trajopt import trajopt, plot_trajs


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


def follow_traj(res):
    global rear_axle_center, rear_axle_theta, rear_axle_velocity, cmd_pub
    rospy.wait_for_message("/ackermann_vehicle/ground_truth/state", Odometry, 5)

    all_trajs = np.concatenate(res['trajs'])
    all_d_trajs = np.concatenate(res['d_trajs'])
    all_angles = np.arctan2(all_d_trajs[:, 1], all_d_trajs[:, 0])
    all_pts = np.c_[all_trajs, all_angles]
    thresh = np.array([0.1, 0.1, np.deg2rad(1)])

    t = 1
    while True:
        current_pt = np.array([rear_axle_center.position.x, rear_axle_center.position.y, rear_axle_theta])

        while t != len(all_pts):
            target_pt = all_pts[t]
            error = target_pt - current_pt
            if not np.all(np.abs(error) < thresh):
                break
        t += 1

        acc = np.linalg.norm(all_d_trajs[t] - all_d_trajs[t - 1])

        target_pt = all_pts[t]
        error = target_pt - current_pt
        if t == len(all_pts) and np.all(np.abs(error) < thresh):
            break
        print('==========', t)

        position_err = error[:2]
        angle_err = error[2]

        speed = np.linalg.norm(position_err) * 0.1
        current_velocity = np.array([rear_axle_velocity.linear.x, rear_axle_velocity.linear.y])
        current_speed = np.linalg.norm(current_velocity)

        speed = np.linalg.norm(all_d_trajs[t])

        beta = np.arcsin(np.tanh(angle_err / current_speed))
        steering_angle = np.arctan(2 * np.tan(beta))
        
        cmd = AckermannDriveStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = "base_link"
        cmd.drive.speed = min(speed, CS['max_vel'])
        cmd.drive.steering_angle = np.clip(steering_angle, -CS['max_steering_angle'], CS['max_steering_angle'])
        cmd.drive.acceleration = np.clip(acc, -CS['max_acc'], CS['max_acc'])

        print(cmd)
        cmd_pub.publish(cmd)
        rospy.wait_for_message("/ackermann_vehicle/ground_truth/state", Odometry, 5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', '-l', type=str, default='outs')
    parser.add_argument('--tag', '-t', type=str, required=True)
    args = parser.parse_args()
    
    savedir = os.path.join(args.logdir, args.tag)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    
    rospy.init_node('controller')
    cmd_pub = rospy.Publisher('/ackermann_vehicle/ackermann_cmd', AckermannDriveStamped, queue_size=10)

    waypoints = np.zeros((CS['num_waypoints'], 3))
    rospy.Subscriber("/ackermann_vehicle/waypoints", PoseArray, waypoint_cb)
    rospy.wait_for_message("/ackermann_vehicle/waypoints", PoseArray, 5)

    rear_axle_center = Pose()
    rear_axle_velocity = Twist()
    rospy.Subscriber("/ackermann_vehicle/ground_truth/state", Odometry, vehicle_state_cb)
    rospy.wait_for_message("/ackermann_vehicle/ground_truth/state", Odometry, 5)

    wps = np.r_[
            [[rear_axle_center.position.x, rear_axle_center.position.y, 
            euler_from_quaternion([rear_axle_center.orientation.x, rear_axle_center.orientation.y, rear_axle_center.orientation.z, rear_axle_center.orientation.w])[2]]], 
            waypoints
        ]
    init_vel = np.array([rear_axle_velocity.linear.x, rear_axle_velocity.linear.y])

    writer = SummaryWriter(log_dir=os.path.join(args.logdir, 'tb', args.tag))
    res = trajopt(wps, writer, init_vel,
        n_pts=100, 
        # dynamics, steering angle, acceleration, speed, traj bounds
        constraint_weights=[500, 50, 10, 1, 100, 100], 
        max_n_opts=200, 
        lr=5e-1
    )

    plot_trajs(wps, res['trajs_init'], res['d_trajs_init'], title='{} | Init'.format(args.tag), save=os.path.join(savedir, 'init.png'))
    plot_trajs(wps, res['trajs'], res['d_trajs'], title='{} | Final'.format(args.tag), save=os.path.join(savedir, 'final.png'))

    follow_traj(res)
