#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
from transforms3d._gohlketransforms import euler_from_quaternion

import numpy as np

class AngleCalculator(Node):

  def __init__(self):
    super().__init__('AngleCalculatorNode')

    # Set the rate
    self.rate = 50.0
    self.dt = 1.0 / self.rate

    # Create the subscribers and publishers
    self.att_pub = self.create_publisher(Vector3, '/uav/sensors/attitude', 1)
    self.imu_sub = self.create_subscription(Imu, "/uav/sensors/filtered_imu", self.imu_callback, 1)

    self.quarternion_pose=(0,0,0,0)

    # Run the node
    self._timer = self.create_timer(self.dt, self.Run)


  # This is the main loop of this class
  def Run(self):
    # Convert quaternion to euler
    euler = euler_from_quaternion(self.quarternion_pose)
    # Publish the Euler Angle
    msg = Vector3()
    msg.x = np.float64(euler[0])
    msg.y = np.float64(euler[1])
    msg.z = np.float64(euler[2])
    self.att_pub.publish(msg)


  # Call back to get the GPS data
  def imu_callback(self, gps_msg):
    # Get the quarternion message
    self.quarternion_pose = (gps_msg.orientation.x,gps_msg.orientation.y,gps_msg.orientation.z,gps_msg.orientation.w)


# Main function
if __name__ == '__main__':
  rclpy.init()
  try:
    rclpy.spin(AngleCalculator())
  except (ExternalShutdownException, KeyboardInterrupt):
    pass
  finally:
    rclpy.try_shutdown()