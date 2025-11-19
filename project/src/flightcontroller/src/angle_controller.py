#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
import time
import numpy as np

from pid_class import PID
from geometry_msgs.msg import Vector3, TwistStamped
from std_msgs.msg import Bool, Float64

from rcl_interfaces.msg import ParameterDescriptor, ParameterType

class AngleController(Node):

  def __init__(self):

    # Allow the simulator to start
    time.sleep(5)

    super().__init__('AngleControllerNode')

    # Getting the PID parameters
    self.declare_parameter('/angle_controller_node/gains/p', 0.1, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
    self.declare_parameter('/angle_controller_node/gains/i', 0.0, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
    self.declare_parameter('/angle_controller_node/gains/d', 0.0, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
    Kp = self.get_parameter('/angle_controller_node/gains/p').get_parameter_value().double_value
    Ki = self.get_parameter('/angle_controller_node/gains/i').get_parameter_value().double_value
    Kd = self.get_parameter('/angle_controller_node/gains/d').get_parameter_value().double_value

    self.get_logger().info(f'Kp: {Kp}, Ki: {Ki}, Kd: {Kd}')

    # Set the rate
    self.rate = 100.0
    self.dt = 1.0 / self.rate

    # Creating the PID's
    self.rpy_PIDS = [PID(Kp, Ki, Kd, self.rate) for _ in range(3)]

    # Create the setpoints
    self.rpy_setpoints = np.zeros(3, dtype=np.float64)
    self.thrust_setpoint = 10

    # Create the current output readings
    self.rpy_readings = np.zeros(3, dtype=np.float64)

    # Check if the drone has been armed
    self.armed = False

    # Create the subscribers
    self.imu_sub = self.create_subscription(Vector3, "/uav/sensors/attitude", self.euler_angle_callback, 1)
    self.att_sub = self.create_subscription(Vector3, "/uav/input/attitude", self.attitude_set_callback, 1)
    self.thrust_sub = self.create_subscription(Float64, "/uav/input/thrust", self.thrust_callback, 1)
    self.armed_sub = self.create_subscription(Bool, "/uav/armed", self.armed_callback, 1)
    self.yaw_sub = self.create_subscription(Float64, "/uav/input/yaw", self.set_yaw_output, 1)

    self.yaw_output = 0.0

    # Create the publishers
    self.rate_pub = self.create_publisher(TwistStamped, '/uav/input/rateThrust', 10)
    
    # Run the control loop
    self._timer = self.create_timer(self.dt * 2, self.ControlLoop)

  # This is the main loop of this class
  def ControlLoop(self):
    # Create the message we are going to send
    msg = TwistStamped()
    msg.header.stamp = self.get_clock().now().to_msg()

    # If the drone has not been armed
    if self.armed == False:
      # Arm the drone
      msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = (0.0, 0.0, 10.0)
      msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = (0.0, 0.0, 0.0)

    # Behave normally (Drone is armed)
    else:
      # Use a PID to calculate the rates you want to publish to maintain an angle
      output = [np.float64(pids.get_output(setp, read)) for pids, setp, read in zip(self.rpy_PIDS, self.rpy_setpoints, self.rpy_readings)]

      # Create the message
      msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = (np.float64(0), np.float64(0), np.float64(self.thrust_setpoint))
      msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = (output[0], output[1], output[2])

    # Publish the message
    self.rate_pub.publish(msg)
    # self.get_logger().info(f'Angle setpoint {self.rpy_setpoints}')

  # This is the callback for the yaw subscriber
  def set_yaw_output(self, msg):
    self.yaw_output = msg.data

  # Save the new attitude readings
  def euler_angle_callback(self, msg):
    self.rpy_readings[0] = msg.x
    self.rpy_readings[1] = msg.y
    self.rpy_readings[2] = msg.z

  # Save the new attitude setpoints
  def attitude_set_callback(self, msg):
    # Dont allow angles greater than 0.5 for x and y
    self.rpy_setpoints[0] = max(min(msg.x, 0.5),-0.5)
    self.rpy_setpoints[1] = max(min(msg.y, 0.5),-0.5)
    self.rpy_setpoints[2] = msg.z

  # Save the new thrust setpoints
  def thrust_callback(self, msg):
    self.thrust_setpoint = msg.data

  # Save whether the drone is armed or not
  def armed_callback(self, msg):
    self.armed = msg.data


# Main function
if __name__ == '__main__':
  rclpy.init()
  try:
    rclpy.spin(AngleController())
  except (ExternalShutdownException, KeyboardInterrupt):
    pass
  finally:
    rclpy.try_shutdown()