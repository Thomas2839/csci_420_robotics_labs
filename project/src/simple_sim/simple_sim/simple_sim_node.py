#!/usr/bin/env python
import copy

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Vector3, PoseStamped, TransformStamped

from transforms3d._gohlketransforms import euler_from_quaternion, quaternion_from_euler
from tf2_ros import TransformBroadcaster

def arr(vec):
    return np.array([vec.x, vec.y, vec.z])

WOBBLE_CONST = np.pi/8

# Create a class which we will use to take keyboard commands and convert them to a position
class SimpleSim(Node):
    # On node initialization
    def __init__(self):
        super().__init__('SimpleSimNode')
        self.gps_pub = self.create_publisher(PoseStamped, "uav/sensors/gps", 1)
        self.vel_sub = self.create_subscription(Vector3, '/uav/input/velocity', self.get_vel, 1)
        self.vel = Vector3()
        self.pos = PoseStamped()
        self.pos.header.frame_id = 'world'
        self.cmd_vel = Vector3()
        self.pos_noise = 0.01
        self.max_acc = 5
        self.rate = 20
        self.dt = 1.0 / self.rate
        self.wobble = 0
        self.wobble_speed = 1
        self.tf_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(self.dt, self.mainloop)

    def get_vel(self, msg):
        self.cmd_vel = msg

    def mainloop(self):
        # Publish the position
        self.pos.header.stamp = self.get_clock().now().to_msg()
        desired_acc = arr(self.cmd_vel) - arr(self.vel)
        mag = np.linalg.norm(desired_acc)
        thrust_perc = 1
        if mag > self.max_acc:
            thrust_perc = self.max_acc / mag
        self.wobble += self.dt * (self.wobble_speed + np.random.normal(scale=WOBBLE_CONST/10))
        pitch = WOBBLE_CONST * np.cos(self.wobble)
        roll = WOBBLE_CONST * np.sin(self.wobble)
        quat = quaternion_from_euler(pitch, roll, 0.0)
        self.pos.pose.orientation.x, self.pos.pose.orientation.y, self.pos.pose.orientation.z, self.pos.pose.orientation.w = quat
        self.vel.x += thrust_perc * (self.cmd_vel.x - self.vel.x)
        self.vel.y += thrust_perc * (self.cmd_vel.y - self.vel.y)
        self.vel.z += thrust_perc * (self.cmd_vel.z - self.vel.z)
        self.pos.pose.position.x += self.vel.x * self.dt
        self.pos.pose.position.y += self.vel.y * self.dt
        self.pos.pose.position.z += self.vel.z * self.dt

        gps_pos = copy.deepcopy(self.pos)
        gps_pos.pose.position.x += np.random.normal(scale=self.pos_noise)
        gps_pos.pose.position.y += np.random.normal(scale=self.pos_noise)
        gps_pos.pose.position.z += np.random.normal(scale=self.pos_noise)
        self.gps_pub.publish(gps_pos)

        t = TransformStamped()
        # header
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'uav_imu'

        # position
        t.transform.translation.x = gps_pos.pose.position.x
        t.transform.translation.y = gps_pos.pose.position.y
        t.transform.translation.z = gps_pos.pose.position.z

        # orientation
        t.transform.rotation.x = self.pos.pose.orientation.x
        t.transform.rotation.y = self.pos.pose.orientation.y
        t.transform.rotation.z = self.pos.pose.orientation.z
        t.transform.rotation.w = self.pos.pose.orientation.w

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    try:
        rclpy.spin(SimpleSim())
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        rclpy.try_shutdown()

# Main function
if __name__ == '__main__':
    main()