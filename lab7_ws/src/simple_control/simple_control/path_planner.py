#!/usr/bin/env python
import copy
import numpy as np
from astar_class import AStarPlanner
from geometry_msgs.msg import Vector3
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Create a class which we will use to take keyboard commands and convert them to a position
class PathPlanner(Node):
    # On node initialization
    def __init__(self):
        super().__init__('path_planner')
        self.origin_x = None
        self.origin_y = None
        # Create the publisher and subscriber
        self.position_pub = self.create_publisher(Vector3, '/uav/input/position_request', 1)
        self.path_pub = self.create_publisher(Int32MultiArray, '/uav/path', 1)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.get_map, qos)
        self.requested_position = self.create_subscription(Vector3, '/uav/input/goal', self.get_goal, 10)
        self.at_waypoint_sub = self.create_subscription(Bool, '/uav/sensors/at_waypoint', self.at_waypoint, 10)

        # Initialize class variables
        self.width = -1
        self.height = -1
        self.drone_position = []
        self.goal_position = []
        self.have_plan = False
        self.map = []
        self.at_waypoint = False

        # Checks if the plan has been started
        self.have_plan = False
        self.sent_position = False

        # Create the path publish message
        self.p_path = Int32MultiArray()
        self.path = None


        # Set the timer to call the mainloop of our class
        self.rate = 5
        self.dt = 1.0 / self.rate
        self.timer = self.create_timer(self.dt, self.mainloop)
        self.get_logger().info('Waiting for goal from tower to map')

    # Map callback
    def get_map(self, msg):

        # Get the map width and height
        self.width = msg.info.width
        self.height = msg.info.height

        # Get the drone position
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        self.drone_position = [int(self.width + self.origin_x), int(self.height + self.origin_y)]

        # Get the map
        self.map = np.reshape(msg.data, (self.width, self.height))

        # Reset plan
        self.have_plan = False

    # Callback for the keyboard manager
    def at_waypoint(self, msg):
        # Save the drones alitude
        self.at_waypoint = msg.data

    # Goal callback
    def get_goal(self, msg):
        if self.origin_x is None or self.origin_y is None:  # map has not been set up yt
            return

        if len(self.goal_position) == 0:
            # Get the goal position
            x = int(round(msg.x, 0) - self.origin_x)
            y = int(round(msg.y, 0) - self.origin_y)

            # Get the drone position
            self.goal_position = [x, y]
            self.get_logger().info(f'Received goal in world coords: {(x, y)}')

            # Reset plan
            self.have_plan = False

    def mainloop(self):
        # Set the rate of this loop
        # If you dont have a plan wait for a map, current position, and a goal
        if not self.have_plan:
            # If we have received the data
            if (len(self.map) != 0) and (len(self.drone_position) == 2) and (len(self.goal_position) == 2):
                self.get_logger().info('Planning path')
                astar = AStarPlanner(safe_distance=1)
                self.path = astar.plan(self.map, self.drone_position, self.goal_position)
                if self.path is not None:
                    self.path = np.array(self.path)
                    self.have_plan = True
                    self.path[:, 0] = self.path[:, 0] + self.origin_x
                    self.path[:, 1] = self.path[:, 1] + self.origin_y
                    self.get_logger().info(f'Executing path: {self.path}')
                else:
                    self.get_logger().info('Path not found, try another goal')
        else: # We have a plan, execute it
            # Publish the current waypoint
            if self.at_waypoint == False or self.sent_position == False or np.shape(self.path)[0] < 0:
                msg = Vector3()
                msg.x = float(self.path[0][0])
                msg.y = float(self.path[0][1])
                msg.z = float(3)
                self.position_pub.publish(msg)
                self.sent_position = True
            else:
                self.path = self.path[1:]
                self.sent_position = False

            # If we are done wait for next goal
            if np.shape(self.path)[0] <= 0 and self.at_waypoint:
                self.have_plan = False
                self.drone_position = copy.deepcopy(self.goal_position)
                self.goal_position = []
                # Publish the path
            if len(self.p_path.data) != len(np.reshape(self.path,-1)):
                self.p_path.data = np.reshape(self.path,-1)
                self.path_pub.publish(self.p_path)



def main():
    rclpy.init()
    try:
        rclpy.spin(PathPlanner())
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        rclpy.try_shutdown()

# Main function
if __name__ == '__main__':
    main()