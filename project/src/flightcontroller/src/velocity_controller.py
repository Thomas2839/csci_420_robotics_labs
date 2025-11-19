#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
import time
import math
import numpy as np

from pid_class import PID
from geometry_msgs.msg import TwistStamped, Vector3
from std_msgs.msg import Float64

from rcl_interfaces.msg import ParameterDescriptor, ParameterType

class VelocityController(Node):

  def __init__(self):

    # Allow the simulator to start
    time.sleep(5)

    super().__init__('VelocityControllerNode')

    # Set the rate
    self.rate = 100.0
    self.dt = 1.0 / self.rate

    # Getting the PID parameters
    self.declare_parameter('/velocity_controller_node/gains/p_xy', 1.0, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
    self.declare_parameter('/velocity_controller_node/gains/i_xy', 0.0, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
    self.declare_parameter('/velocity_controller_node/gains/d_xy', 0.0, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
    self.declare_parameter('/velocity_controller_node/gains/p_z', 1.0, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
    self.declare_parameter('/velocity_controller_node/gains/i_z', 0.0, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
    self.declare_parameter('/velocity_controller_node/gains/d_z', 0.0, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
    Kp_xy = self.get_parameter('/velocity_controller_node/gains/p_xy').get_parameter_value().double_value
    Ki_xy = self.get_parameter('/velocity_controller_node/gains/i_xy').get_parameter_value().double_value
    Kd_xy = self.get_parameter('/velocity_controller_node/gains/d_xy').get_parameter_value().double_value
    Kp_z = self.get_parameter('/velocity_controller_node/gains/p_z').get_parameter_value().double_value
    Ki_z = self.get_parameter('/velocity_controller_node/gains/i_z').get_parameter_value().double_value
    Kd_z = self.get_parameter('/velocity_controller_node/gains/d_z').get_parameter_value().double_value

    self.get_logger().info(f'{Kp_xy}, {Ki_xy}, {Kd_xy}, {Kp_z}, {Ki_z}, {Kd_z}')

    # Creating the PID's
    self.vel_PIDs = [PID(Kp_xy, Ki_xy, Kd_xy, self.rate) for _ in range(2)]
    self.vel_PIDs.append(PID(Kp_z, Ki_z, Kd_z, self.rate))

    # Get the setpoints
    self.setpoints = np.zeros(3, dtype=np.float64)

    # Create the current output readings
    self.vel_readings = np.zeros(3, dtype=np.float64)

    # Reading the yaw
    self.yaw_reading = 0

    # Create the subscribers
    self.vel_read_sub = self.create_subscription(TwistStamped, "/uav/sensors/velocity", self.get_vel, 1)
    self.vel_set_sub = self.create_subscription(Vector3, '/uav/input/velocity', self.set_vel, 1)
    self.sub = self.create_subscription(Vector3, '/uav/sensors/attitude', self.euler_angle_callback, 1)

    # Create the publishers 
    self.att_pub = self.create_publisher(Vector3, "/uav/input/attitude", 1)
    self.thrust_pub = self.create_publisher(Float64, '/uav/input/thrust', 1)

    # Data we will be publishing
    self.z_output = Float64()
    self.z_output.data = 0.0

    # Run the control loop
    self._timer = self.create_timer(self.dt * 2, self.ControlLoop)

  # This is the main loop of this class
  def ControlLoop(self):
    # Use a PID to calculate the angle you want to hold and thrust you want
    output = [pids.get_output(setp, vel) for pids, setp, vel in zip(self.vel_PIDs, self.setpoints, self.vel_readings)]

    # Transform X and Y velocity from world frame to drone frame (i.e. yaw needs to be taken into account)
    x_output =   math.cos(self.yaw_reading) * output[1] - math.sin(self.yaw_reading) * output[0]
    y_output = - math.sin(self.yaw_reading) * output[1] - math.cos(self.yaw_reading) * output[0]

    # Set the yaw at 0
    yaw = 0

    # The z output is the PID + gravity
    self.z_output.data = output[2] + 9.8

    # Limit the thrust to the drone
    self.z_output.data = np.float64(min(max(0, self.z_output.data), 20))

    # Create and publish the data (0 yaw)
    attitude = Vector3()
    attitude.x, attitude.y, attitude.z = (np.float64(x_output), np.float64(y_output), np.float64(yaw))
    self.att_pub.publish(attitude)
    self.thrust_pub.publish(self.z_output)
    # self.get_logger().info(f'Velocity setpoint {self.setpoints}')


  # Call back to get the velocity data
  def get_vel(self, vel_msg):
    self.vel_readings[0] = vel_msg.twist.linear.x
    self.vel_readings[1] = vel_msg.twist.linear.y
    self.vel_readings[2] = vel_msg.twist.linear.z

  # Get the yaw reading
  def euler_angle_callback(self, msg):
    self.yaw_reading = msg.z

  # Call back to get the velocity setpoints
  def set_vel(self, vel_msg):
    self.setpoints[0] = vel_msg.x
    self.setpoints[1] = vel_msg.y
    self.setpoints[2] = vel_msg.z


# Main function
if __name__ == '__main__':
  rclpy.init()
  try:
    rclpy.spin(VelocityController())
  except (ExternalShutdownException, KeyboardInterrupt):
    pass
  finally:
    rclpy.try_shutdown()