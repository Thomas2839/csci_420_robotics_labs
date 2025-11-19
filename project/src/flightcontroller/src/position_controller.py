#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
import time
import numpy as np

from pid_class import PID
from geometry_msgs.msg import Vector3, PoseStamped

from rcl_interfaces.msg import ParameterDescriptor, ParameterType

class PositionController(Node):

  def __init__(self):

    # Allow the simulator to start
    time.sleep(5)

    super().__init__('PositionControllerNode')

    # Set the rate
    self.rate = 100.0
    self.dt = 1.0 / self.rate

    # Getting the PID parameters
    self.declare_parameter('/position_controller_node/gains/stable/p', 1.0, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
    self.declare_parameter('/position_controller_node/gains/stable/i', 0.0, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
    self.declare_parameter('/position_controller_node/gains/stable/d', 0.0, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
    Kp = self.get_parameter('/position_controller_node/gains/stable/p').get_parameter_value().double_value
    Ki = self.get_parameter('/position_controller_node/gains/stable/i').get_parameter_value().double_value
    Kd = self.get_parameter('/position_controller_node/gains/stable/d').get_parameter_value().double_value

    self.get_logger().info(f'Kp: {Kp}, Ki: {Ki}, Kd: {Kd}')

    # Creating the PID's
    self.position_PIDs = [PID(Kp, Ki, Kd, self.rate) for _ in range(3)]

    # Get the setpoints
    self.setpoints = np.array([0.0, 0.0, 3.0], dtype=np.float64)

    # Create the current output readings
    self.current_position = np.zeros(3, dtype=np.float64)

    # Create the subscribers
    self.gps_sub = self.create_subscription(PoseStamped, "uav/sensors/gps", self.get_gps, 1)
    self.pos_set_sub = self.create_subscription(Vector3, "uav/input/position", self.set_pos, 1)

    # Create the publishers
    self.vel_set_sub = self.create_publisher(Vector3, '/uav/input/velocity', 1)

    # Run the control loop
    self._timer = self.create_timer(self.dt * 2, self.ControlLoop)

  # This is the main loop of this class
  def ControlLoop(self):
    # Use a PID to calculate the velocity you want
    output = [np.float64(pids.get_output(setp, pos)) for pids, setp, pos in zip(self.position_PIDs, self.setpoints, self.current_position)]

    # Create and publish the data
    velocity = Vector3()
    velocity.x, velocity.y, velocity.z = (output[0], output[1], output[2])
    self.vel_set_sub.publish(velocity)
    # self.get_logger().info(f'Position setpoint {self.setpoints}')

  # Call back to get the gps data
  def get_gps(self, msg):
    self.current_position[0] = msg.pose.position.x
    self.current_position[1] = msg.pose.position.y
    self.current_position[2] = msg.pose.position.z


  # Call back to get the position setpoints
  def set_pos(self, msg):
    # If our set point changes reset the PID build up
    check_x = abs(self.setpoints[0] - msg.x) > 1e-6
    check_y = abs(self.setpoints[1] - msg.y) > 1e-6
    check_z = abs(self.setpoints[2] - msg.z) > 1e-6
    # Remove the buildup
    if check_x or check_y or check_z:
      for pid in self.position_PIDs:
        pid.remove_buildup()
    # Set new setpoint
    self.setpoints[0] = msg.x
    self.setpoints[1] = msg.y
    self.setpoints[2] = msg.z


# Main function
if __name__ == '__main__':
  rclpy.init()
  try:
    rclpy.spin(PositionController())
  except (ExternalShutdownException, KeyboardInterrupt):
    pass
  finally:
    rclpy.try_shutdown()