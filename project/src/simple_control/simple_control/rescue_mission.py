#!/usr/bin/env python3
"""Rescue mission node (final project)

This builds on the earlier checkpoint and adds the components
required for the final checkpoint:
- obtain the dog's location from /cell_tower/position (using TF)
- plan a global path from the drone to the dog using A*
- follow that path using /uav/input/position
- detect doors blocking the path and attempt to open them using the use_key service
- publish the final shortest path on /uav/final_path

"""

from __future__ import annotations

import math
import heapq
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import PoseStamped, Point, Vector3, PointStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32, Int32MultiArray

from environment_controller.srv import UseKey

import tf2_ros
from tf2_ros import TransformException
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point


class RescueMission(Node):
    """Combined mapping + planning node.
    New additions include:
    - TF conversion of dog position
    - A* path planning
    - Simple mission state machine
    - Basic local check for doors
    - Movement commands via /uav/input/position
    """

    # simple mission states
    WAIT_FOR_GOAL = 0
    PLAN_PATH = 1
    FOLLOW_PATH = 2
    AT_GOAL = 3

    def __init__(self):
        super().__init__('rescue_mission_node')

        # ============================
        # MAPPING SETUP
        # ============================

        # Map parameters
        default_w = 23
        default_h = 23
        self.declare_parameter('map_width', default_w)
        self.declare_parameter('map_height', default_h)

        self.map_width = int(self.get_parameter('map_width').value)
        self.map_height = int(self.get_parameter('map_height').value)

        # We use 1 meter per cell (resolution 1.0)
        self.resolution = 1.0

        # Map origin: bottom-left in world coordinates; drone starts at (0,0)
        # Shift the origin by +0.5 so cell centers align with integer
        self.origin_x = -self.map_width / 2.0 + 0.5
        self.origin_y = -self.map_height / 2.0 + 0.5

        # Occupancy grid stored as flat list in row-major (x,y) to match GUI
        self.grid = [50] * (self.map_width * self.map_height)

        # publishers
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 1)
        self.cmd_pub = self.create_publisher(Vector3, "/uav/input/position", 10)
        self.final_path_pub = self.create_publisher(Int32MultiArray, "/uav/final_path", 1)
        self.debug_path_pub = self.create_publisher(Int32MultiArray, "/uav/path", 1)

        # subscribers
        self.gps_sub = self.create_subscription(PoseStamped, '/uav/sensors/gps', self.gps_callback, 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/uav/sensors/lidar', self.lidar_callback, 10)
        self.key_sub = self.create_subscription(Int32, "/keys_remaining", self.keys_cb, 10)
        self.cell_sub = self.create_subscription(PointStamped, "/cell_tower/position", self.cell_tower_cb, 10)

        # TF buffer/listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # door-opening service
        self.use_key_client = self.create_client(UseKey, 'use_key')

        # internal state
        self.has_gps = False
        self.drone_x = 0.0
        self.drone_y = 0.0

        self.keys_remaining = 4
        self.has_goal = False
        self.goal_world = None
        self.goal_cell = None

        self.current_path_cells = []
        self.current_path_world = []
        self.wp_index = 0
        self.tried_door_cells = set()

        # mission state
        self.state = self.WAIT_FOR_GOAL

        # periodic publishing + mission loop
        self.create_timer(0.5, self.publish_map)
        self.create_timer(0.2, self.mission_step)

        # mark start cell free
        sx, sy = self.world_to_map(0.0, 0.0)
        self.set_cell(sx, sy, 0)

        self.get_logger().info(f"Rescue mission initialized. Map {self.map_width}x{self.map_height}")

    # ============================
    # Coordinate conversions
    # ============================

    def world_to_map(self, wx, wy):
        mx = int(math.floor((wx - self.origin_x) + 0.5))
        my = int(math.floor((wy - self.origin_y) + 0.5))
        return mx, my

    def map_to_world(self, mx, my):
        wx = self.origin_x + mx * self.resolution
        wy = self.origin_y + my * self.resolution
        return float(wx), float(wy)

    def map_index(self, mx, my):
        if mx < 0 or my < 0 or mx >= self.map_width or my >= self.map_height:
            return -1
        return my + mx * self.map_height

    def get_cell(self, mx, my):
        idx = self.map_index(mx, my)
        if idx < 0:
            return None
        return self.grid[idx]

    def set_cell(self, mx, my, val):
        idx = self.map_index(mx, my)
        if idx < 0:
            return
        cur = self.grid[idx]
        if cur < 0:  # preserve doors/goal
            return
        if cur == 0 and val > 0:  # sticky free rule
            return
        self.grid[idx] = max(0, min(100, int(val)))

    # ============================
    # Callbacks
    # ============================

    def gps_callback(self, msg):
        self.drone_x = msg.pose.position.x
        self.drone_y = msg.pose.position.y
        self.has_gps = True

    def keys_cb(self, msg):
        self.keys_remaining = msg.data

    def cell_tower_cb(self, msg):
        """Get dog's last known position in cell_tower frame → transform to world."""
        if self.has_goal:
            return

        try:
            transform = self.tf_buffer.lookup_transform("world", msg.header.frame_id, Time().to_msg())
            world_pt = do_transform_point(msg, transform)

            gx = world_pt.point.x
            gy = world_pt.point.y

            self.goal_world = (gx, gy)
            self.goal_cell = self.world_to_map(gx, gy)

            mx, my = self.goal_cell
            idx = self.map_index(mx, my)
            if idx >= 0:
                self.grid[idx] = -3  # goal marker

            self.has_goal = True
            self.get_logger().info(f"Dog located at world=({gx:.2f},{gy:.2f}), map={self.goal_cell}")

        except TransformException as e:
            self.get_logger().warn(f"TF failed: {e}")

    # ============================
    # LIDAR mapping
    # ============================

    def lidar_callback(self, msg):
        if not self.has_gps:
            return

        angle = msg.angle_min
        for r in msg.ranges:
            if math.isfinite(r) and r <= 0:
                angle += msg.angle_increment
                continue

            max_range = msg.range_max if math.isfinite(msg.range_max) else 5.0
            dist = r if math.isfinite(r) else max_range

            step = max(0.1, self.resolution / 4.0)
            t = 0.0
            last = None

            while t < dist:
                wx = self.drone_x + t * math.cos(angle)
                wy = self.drone_y + t * math.sin(angle)
                mx, my = self.world_to_map(wx, wy)

                if (mx, my) != last:
                    cur = self.get_cell(mx, my)
                    if cur not in (-1, -2, -3) and cur is not None:
                        self.set_cell(mx, my, cur - 15)
                    last = (mx, my)

                t += step

            if math.isfinite(r) and dist > 0:
                hx = self.drone_x + dist * math.cos(angle)
                hy = self.drone_y + dist * math.sin(angle)
                mx, my = self.world_to_map(hx, hy)
                cur = self.get_cell(mx, my)
                if cur is not None:
                    self.set_cell(mx, my, cur + 30)

            angle += msg.angle_increment

    # ============================
    # Mission step machine
    # ============================

    def mission_step(self):
        if self.state == self.WAIT_FOR_GOAL:
            if self.has_gps and self.has_goal:
                self.get_logger().info("Goal + GPS acquired → planning.")
                self.state = self.PLAN_PATH

        elif self.state == self.PLAN_PATH:
            self.plan_path()

        elif self.state == self.FOLLOW_PATH:
            self.follow_path()

        elif self.state == self.AT_GOAL:
            return

    # ============================
    # A* Planning
    # ============================

    def is_free(self, mx, my):
        val = self.get_cell(mx, my)
        if val is None:
            return False
        if val == -1:
            return False
        if val in (-2, -3):
            return True
        return val < 70

    def neighbors(self, mx, my):
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = mx + dx, my + dy
            if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                if self.is_free(nx, ny):
                    yield (nx, ny)

    def astar(self, start, goal):
        def h(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

        pq = [(h(start,goal), start)]
        g = {start: 0}
        came = {start: None}

        while pq:
            _, cur = heapq.heappop(pq)

            if cur == goal:
                path = []
                c = cur
                while c:
                    path.append(c)
                    c = came[c]
                return path[::-1]

            for nbr in self.neighbors(*cur):
                cell_val = self.get_cell(*nbr)
                step = 1.0 + (0.2 if cell_val == 50 else 0)
                new_g = g[cur] + step

                if new_g < g.get(nbr, 1e9):
                    g[nbr] = new_g
                    came[nbr] = cur
                    heapq.heappush(pq, (new_g + h(nbr,goal), nbr))

        return []

    def plan_path(self):
        start = self.world_to_map(self.drone_x, self.drone_y)
        goal = self.goal_cell

        self.get_logger().info(f"Planning path from {start} → {goal}...")
        path = self.astar(start, goal)

        if not path:
            self.get_logger().warn("A* failed → will retry.")
            return

        self.current_path_cells = path
        self.current_path_world = [self.map_to_world(mx, my) for (mx,my) in path]
        self.wp_index = 0

        # debug path publish
        flat = []
        for mx, my in path:
            flat.extend([mx, my])
        msg = Int32MultiArray(data=flat)
        self.debug_path_pub.publish(msg)

        self.state = self.FOLLOW_PATH

    # ============================
    # Path following + door handling
    # ============================

    def follow_path(self):
        if self.wp_index >= len(self.current_path_world):
            self.get_logger().info("Reached the goal. Mission complete!")
            self.publish_final_path()
            self.state = self.AT_GOAL
            return

        wx, wy = self.current_path_world[self.wp_index]
        mx, my = self.world_to_map(wx, wy)
        cell = self.get_cell(mx, my)

        # blocked → try door or replan
        if cell is not None and cell >= 70:
            # try opening if close enough
            if self.keys_remaining > 0 and (mx, my) not in self.tried_door_cells:
                dist = math.hypot(wx - self.drone_x, wy - self.drone_y)
                if dist <= 1.0:
                    self.get_logger().info(f"Attempting to open door at {mx,my}")
                    self.tried_door_cells.add((mx,my))
                    self.try_open_door(wx, wy, mx, my)
                    return

            # otherwise replan
            self.state = self.PLAN_PATH
            return

        # otherwise move to waypoint
        cmd = Vector3(x=float(wx), y=float(wy), z=0.0)
        self.cmd_pub.publish(cmd)

        # check arrival
        if math.hypot(self.drone_x - wx, self.drone_y - wy) < 0.4:
            self.wp_index += 1

    def try_open_door(self, wx, wy, mx, my):
        if not self.use_key_client.service_is_ready():
            self.get_logger().warn("use_key service not ready.")
            return

        req = UseKey.Request()
        req.door_loc = Point(x=float(wx), y=float(wy), z=0.0)

        future = self.use_key_client.call_async(req)

        def callback(fut, cell=(mx,my)):
            try:
                res = fut.result()
            except Exception as e:
                self.get_logger().warn(f"use_key failed: {e}")
                return

            cmx, cmy = cell
            idx = self.map_index(cmx, cmy)
            if idx < 0: return

            if res.success:
                self.grid[idx] = -2  # opened door
                self.get_logger().info(f"Door opened at {cell}")
            else:
                self.grid[idx] = -1  # closed door
                self.get_logger().info(f"Door failed at {cell}")

        future.add_done_callback(callback)

    # ============================
    # Publish map + final path
    # ============================

    def publish_map(self):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        info = MapMetaData()
        info.resolution = self.resolution
        info.width = self.map_width
        info.height = self.map_height
        info.origin.position.x = self.origin_x
        info.origin.position.y = self.origin_y
        msg.info = info

        msg.data = [int(v) for v in self.grid]
        self.map_pub.publish(msg)

    def publish_final_path(self):
        flat = []
        for mx, my in self.current_path_cells:
            flat.extend([mx, my])
        self.final_path_pub.publish(Int32MultiArray(data=flat))
        self.get_logger().info("Published final path.")


def main(args=None):
    rclpy.init(args=args)
    node = RescueMission()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
