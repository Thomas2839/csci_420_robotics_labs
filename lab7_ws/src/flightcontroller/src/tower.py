#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
import time
import copy

import numpy as np

from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Vector3
from std_msgs.msg import Int32MultiArray
from nav_msgs.msg import OccupancyGrid
from transforms3d._gohlketransforms import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import traceback
class Tower(Node):

    def __init__(self):

        time.sleep(15)
        super().__init__('TowerNode')

        # Goal publisher
        self.goal_pub = self.create_publisher(Vector3, "/tower/goal", 1)

        # map variables
        self.map = None
        self.map_origin = None
        self.empty_traj = True
        self.has_new_goal = False

        # subscribers subscriber
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.get_map, qos)
        self.traj_sub = self.create_subscription(Int32MultiArray, '/uav/path', self.handle_traj, 1)

        # Transform
        # This is the transfrom from tester node
        self.towertransform = TransformStamped()
        self.towertransform.transform.translation.x = -128.5751
        self.towertransform.transform.translation.y = 77.25571
        self.towertransform.transform.translation.z = float(0)

        quat = quaternion_from_euler(float(0), float(0), float(+0.523599))
        self.towertransform.transform.rotation.x = quat[1]
        self.towertransform.transform.rotation.y = quat[2]
        self.towertransform.transform.rotation.z = quat[3]
        self.towertransform.transform.rotation.w = quat[0]

        self.towertransform.header.frame_id = 'tower'
        self.towertransform.child_frame_id = 'world'

        self.cur_goal = None

        # start main loop
        self.rate = 10
        self.dt = 1.0 / self.rate
        self.timer = self.create_timer(self.dt, self.mainloop)

    def handle_traj(self, msg):
        if len(msg.data) == 0:
            self.empty_traj = True
        else:
            self.empty_traj = False
            self.has_new_goal = False

    # Map callback
    def get_map(self, msg):
        # Get map transform
        self.map_origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y])
        # Get the map
        self.map = np.reshape(msg.data, (msg.info.width, msg.info.height))
        self.map = self.expand_obstacles(self.map, 1)

    def generate_goal(self):
        if self.map is None:
            return
        try:
            MAX_COUNT = self.map.shape[0] * self.map.shape[1]
            count = 0
            while count < MAX_COUNT:
                # choose a random position
                i = np.random.randint(0, self.map.shape[0])
                j = np.random.randint(0, self.map.shape[1])
                if self.map[i][j] == 0:
                    return [i + self.map_origin[0], j + self.map_origin[1]]
                count += 1
            return None
        except:
            traceback.print_exc()
            return None

    def mainloop(self):
        if self.empty_traj and not self.has_new_goal:
            new_goal = self.generate_goal()
            if new_goal:
                point = PointStamped()
                point.point.x = float(new_goal[0])
                point.point.y = float(new_goal[1])
                point.point.z = 0.0
                new_point = do_transform_point(point, self.towertransform)
                self.get_logger().info(f'Published goal [{new_point.point.x}, {new_point.point.y}]')
                msg = Vector3()
                msg.x, msg.y, msg.z = float(new_point.point.x), float(new_point.point.y), 0.0
                self.cur_goal = msg
                self.has_new_goal = True
        if self.empty_traj and self.cur_goal:
            self.goal_pub.publish(self.cur_goal)

    # Expand the obstacles by distance so you do not hit one
    def expand_obstacles(self, map_data, distance):
        new_map = copy.deepcopy(map_data)
        # For each element in the map
        for i in range(map_data.shape[0]):
            for j in range(map_data.shape[1]):
                # if this is an obstacle
                if map_data[i, j] != 0:
                    # Expand the obstacle by 1 in all directions
                    for x_dim in range((-1 * distance), (1 * distance) + 1):
                        for y_dim in range((-1 * distance), (1 * distance) + 1):
                            if (0 <= i + x_dim < new_map.shape[0]) and (0 <= j + y_dim < new_map.shape[1]):
                                new_map[i + x_dim, j + y_dim] = map_data[i, j]
        return new_map


def main():
    rclpy.init()
    try:
        rclpy.spin(Tower())
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        rclpy.try_shutdown()

# Main function
if __name__ == '__main__':
    main()