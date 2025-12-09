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
import time
import heapq
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import PoseStamped, Vector3, PointStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32, Int32MultiArray

from environment_controller.srv import UseKey

import numpy as np
from .door_detector import DoorDetector

import tf2_ros
from tf2_ros import TransformException
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from simple_control.astar_class import AStarPlanner 


class RescueMission(Node):
    """Combined mapping + planning node with explicit mission states."""

    # ============================
    # Mission states (your design)
    # ============================
    EXPLORING_WORLD = 0
    LOCATING_DOORS = 1
    OPENING_DOORS = 2
    MOVING_TO_WAYPOINT = 3
    UPDATING_MAP = 4
    AT_GOAL = 5

    def __init__(self):
        super().__init__('rescue_mission')
        # ============================
        # MAPPING SETUP
        # ============================

        default_w = 23   
        default_h = 23
        self.declare_parameter('map_width', default_w)
        self.declare_parameter('map_height', default_h)

        self.map_width = int(self.get_parameter('map_width').value)
        self.map_height = int(self.get_parameter('map_height').value)

        self.resolution = None

        # map origin in world frame (so cell centers line up with integers)
        self.origin_x = -self.map_width / 2.0 + 0.5
        self.origin_y = -self.map_height / 2.0 + 0.5

        # Occupancy grid: row-major
        self.grid = [50] * (self.map_width * self.map_height)

        self.map_ready = False

        # Publishers
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 1)
        self.cmd_pub = self.create_publisher(Vector3, "/uav/input/position", 10)
        self.final_path_pub = self.create_publisher(Int32MultiArray, "/uav/final_path", 1)
        self.debug_path_pub = self.create_publisher(Int32MultiArray, "/uav/path", 1)

        # Subscribers
        self.gps_sub = self.create_subscription(
            PoseStamped, '/uav/sensors/gps', self.gps_callback, 10
        )
        self.lidar_sub = self.create_subscription(
            LaserScan, '/uav/sensors/lidar', self.lidar_callback, 10
        )
        self.key_sub = self.create_subscription(
            Int32, "/keys_remaining", self.keys_cb, 10
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid,"/map",self.map_callback, 10
        )

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Services
        self.use_key_client = self.create_client(UseKey, 'use_key')

        # Internal state
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

        # doors
        self.door_detector = DoorDetector(
            buffer_size=6, sample_interval=0.8, std_multiplier=2.0
        )
        self.detected_doors = set()       # set of (mx, my) cells
        self.tried_door_cells = set()     # doors we've already tried
        self.current_blocked_cell = None  # where we discovered an obstruction
        self.current_door_cell = None     # door we're targeting now
        self.pending_door_future = None

        # Mission state machine
        self.state = self.EXPLORING_WORLD

        # Timers
        self.create_timer(0.5, self.publish_map)
        self.create_timer(0.2, self.mission_step)
        self.create_timer(0.5, self.update_goal_from_tf)

        # mark start cell free
        sx, sy = self.world_to_map(0.0, 0.0)
        self.set_cell(sx, sy, 0)

        self.get_logger().info(
            f"Rescue mission initialized. Map {self.map_width}x{self.map_height}, "
            f"starting in EXPLORING_WORLD"
        )

    # ============================
    # Coordinate conversions
    # ============================

    def world_to_map(self, wx, wy):
        mx = int(math.floor((wx - self.origin_x) / self.resolution))
        my = int(math.floor((wy - self.origin_y) / self.resolution))
        return mx, my

    def map_to_world(self, mx, my):
        wx = self.origin_x + (mx + 0.5) * self.resolution
        wy = self.origin_y + (my + 0.5) * self.resolution
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
        # preserve sentinel values
        if cur < 0:
            return
        # don't overwrite definitely free with "less free"
        if cur == 0 and val > 0:
            return
        self.grid[idx] = max(0, min(100, int(val)))

    # ============================
    # Callbacks
    # ============================

    def gps_callback(self, msg: PoseStamped):
        self.drone_x = msg.pose.position.x
        self.drone_y = msg.pose.position.y
        self.has_gps = True

    def keys_cb(self, msg: Int32):
        self.keys_remaining = msg.data

    def map_callback(self, msg):
        self.map = msg
        self.map_ready = True

        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.resolution = msg.info.resolution

        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y

        # Convert map data from int8 array to Python list
        self.grid = list(msg.data)

        self.get_logger().info(
            f"Map loaded: {self.map_width}x{self.map_height}, origin=({self.origin_x},{self.origin_y}), res={self.resolution}"
        )

    def update_goal_from_tf(self):
        if self.has_goal:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                "cell_tower", "world", rclpy.time.Time()
            )

            gx = transform.transform.translation.x
            gy = transform.transform.translation.y

            self.get_logger().info(
                f"[TF-GOAL] Dog located via TF at tower=({gx:.2f}, {gy:.2f})"
            )

            mx, my = self.world_to_map(gx, gy)
            # Hardcode for debugging:
            self.goal = (10, 10)  # map coordinates

            self.get_logger().info(f"Debug goal set to {self.goal}")
            self.goal_cell = (10, 10)
            self.has_goal = True

            idx = self.map_index(mx, my)
            if idx >= 0:
                self.grid[idx] = -3

            self.get_logger().info(
                f"[TF-GOAL] Mapped dog to cell=({mx},{my}) — goal set."
            )

        except Exception as e:
            # Normal until TF has enough data
            self.get_logger().debug(f"[TF-GOAL] TF not ready: {e}")


    def lidar_callback(self, msg: LaserScan):
        # Mapping happens regardless of mission state
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

            # free cells along beam
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

            # occupied at hit
            if math.isfinite(r) and dist > 0:
                hx = self.drone_x + dist * math.cos(angle)
                hy = self.drone_y + dist * math.sin(angle)
                mx, my = self.world_to_map(hx, hy)
                cur = self.get_cell(mx, my)
                if cur is not None:
                    self.set_cell(mx, my, cur + 30)

            angle += msg.angle_increment

        # Door detection (runs slower internally via DoorDetector)
        try:
            sampled = self.door_detector.sample_scan(msg)
            if sampled:
                candidates = self.door_detector.detect_standouts()
                for beam_idx, stdv in candidates:
                    ang = msg.angle_min + beam_idx * msg.angle_increment
                    r = self.door_detector.latest_range_for(beam_idx)
                    if not math.isfinite(r):
                        r = msg.range_max if math.isfinite(msg.range_max) else 5.0

                    hx = self.drone_x + r * math.cos(ang)
                    hy = self.drone_y + r * math.sin(ang)
                    mx, my = self.world_to_map(hx, hy)

                    drone_mx, drone_my = self.world_to_map(self.drone_x, self.drone_y)
                    dist_to_drone = math.hypot(mx - drone_mx, my - drone_my)
                    if dist_to_drone < 0.5:
                        continue

                    cur = self.get_cell(mx, my)
                    if cur is not None and cur not in (-1, -2, -3):
                        # avoid duplicating the same door in slightly different cells
                        is_new_door = True
                        for ddx, ddy in self.detected_doors:
                            if math.hypot(mx - ddx, my - ddy) < 1.5:
                                is_new_door = False
                                break

                        if is_new_door:
                            idx = self.map_index(mx, my)
                            if idx >= 0:
                                self.grid[idx] = -1  # door (closed) sentinel
                            self.detected_doors.add((mx, my))

                            if self.door_detector.should_report_candidate(
                                mx, my, report_cooldown=5.0
                            ):
                                self.get_logger().info(
                                    f"New door detected at ({mx}, {my}) "
                                    f"from beam {beam_idx} (std={stdv:.3f})"
                                )
        except Exception as e:
            self.get_logger().warn(f"Door detection error: {e}")

    # ============================
    # Mission State Machine
    # ============================

    def mission_step(self):
        self.get_logger().info(
            f"[STEP] State={self.state} | GPS={self.has_gps} "
            f"| Goal={self.goal_cell} | Keys={self.keys_remaining}"
        )
        # Need both GPS and goal to meaningfully move
        if not self.has_gps or not self.has_goal:
            return

        if self.state == self.EXPLORING_WORLD:
            # Exploring means: build map and, once we know the goal, plan a path.
            if not self.current_path_world:
                self.get_logger().info(
                    "EXPLORING_WORLD: goal known, planning initial path."
                )
                self.plan_path()
            self.state = self.MOVING_TO_WAYPOINT

        elif self.state == self.MOVING_TO_WAYPOINT:
            self.follow_path_step()

        elif self.state == self.LOCATING_DOORS:
            self.locate_door_step()

        elif self.state == self.OPENING_DOORS:
            self.open_door_step()

        elif self.state == self.UPDATING_MAP:
            self.update_map_and_replan()

        elif self.state == self.AT_GOAL:
            # nothing to do
            return

    # ============================
    # A* Planning
    # ============================

    def is_free(self, mx, my):
        val = self.get_cell(mx, my)
        if val is None:
            return False
        if val == -1:   # closed door cell (treated as obstacle until opened)
            return False
        if val in (-2, -3):  # open door or goal
            return True
        return val < 70

    def neighbors(self, mx, my):
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = mx + dx, my + dy
            if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                if self.is_free(nx, ny):
                    yield (nx, ny)

    def astar(self, start, goal):
        def h(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        pq = [(h(start, goal), start)]
        g = {start: 0.0}
        came = {start: None}

        while pq:
            _, cur = heapq.heappop(pq)

            if cur == goal:
                path = []
                c = cur
                while c is not None:
                    path.append(c)
                    c = came[c]
                return path[::-1]

            for nbr in self.neighbors(*cur):
                cell_val = self.get_cell(*nbr)
                step = 1.0 + (0.2 if cell_val == 50 else 0.0)
                new_g = g[cur] + step

                if new_g < g.get(nbr, 1e9):
                    g[nbr] = new_g
                    came[nbr] = cur
                    heapq.heappush(pq, (new_g + h(nbr, goal), nbr))

        return []

    def plan_path(self):
        start = self.world_to_map(self.drone_x, self.drone_y)
        goal = self.goal_cell

        self.get_logger().info(f"Planning path from {start} → {goal}...")
        path = self.astar(start, goal)

        if not path:
            self.get_logger().warn("A* failed → no path currently, will retry later.")
            self.current_path_cells = []
            self.current_path_world = []
            self.wp_index = 0
            return

        self.current_path_cells = path
        self.current_path_world = [self.map_to_world(mx, my) for (mx, my) in path]
        self.wp_index = 0
        self.get_logger().info(
            f"[PLAN] Path length={len(path)} | First WP={path[0] if path else None}"
        )

        # debug path publish
        flat = []
        for mx, my in path:
            flat.extend([mx, my])
        msg = Int32MultiArray(data=flat)
        self.debug_path_pub.publish(msg)

        self.get_logger().info(
            f"New path planned with {len(self.current_path_world)} waypoints."
        )

    # ============================
    # MOVING_TO_WAYPOINT
    # ============================

    def follow_path_step(self):
        # High-level movement debug
        self.get_logger().info(
            f"[MOVE] State=MOVING_TO_WAYPOINT | wp_index={self.wp_index}/{len(self.current_path_world)} "
            f"| drone=({self.drone_x:.2f},{self.drone_y:.2f})"
        )

        # No path or finished path
        if not self.current_path_world:
            self.get_logger().warn("[MOVE] No path available — cannot move. Waiting for replanning.")
            return

        if self.wp_index >= len(self.current_path_world):
            self.get_logger().info("[MOVE] Final waypoint reached → AT_GOAL")
            self.publish_final_path()
            self.state = self.AT_GOAL
            return

        # Extract waypoint
        wx, wy = self.current_path_world[self.wp_index]
        mx, my = self.world_to_map(wx, wy)
        cell = self.get_cell(mx, my)

        self.get_logger().info(
            f"[MOVE] Target WP#{self.wp_index}: world=({wx:.2f},{wy:.2f}) map={mx,my} cell_val={cell}"
        )

        # Check for obstacle at waypoint
        if cell is not None and cell >= 70:
            self.get_logger().warn(
                f"[BLOCK] Path blocked at map cell {mx,my} (val={cell}) → switching to LOCATING_DOORS"
            )
            self.current_blocked_cell = (mx, my)
            self.state = self.LOCATING_DOORS
            return

        # NORMAL MOVEMENT
        self.get_logger().info(
            f"[MOVE] Publishing movement command → ({wx:.2f}, {wy:.2f})"
        )
        cmd = Vector3(x=float(wx), y=float(wy), z=0.0)
        self.cmd_pub.publish(cmd)

        # ARRIVAL CHECK
        dist = math.hypot(self.drone_x - wx, self.drone_y - wy)
        self.get_logger().info(f"[MOVE] Distance to WP = {dist:.3f}")

        if dist < 0.4:
            self.get_logger().info(f"[ARRIVAL] Arrived at WP#{self.wp_index}")

            # Door sealing logic
            if cell == -2:
                idx = self.map_index(mx, my)
                if idx >= 0:
                    self.grid[idx] = 100
                    self.get_logger().info(
                        f"[DOOR] Sealing door behind drone at cell {mx,my} (now marked wall)"
                    )

            self.wp_index += 1
            self.get_logger().info(f"[MOVE] Advancing to WP#{self.wp_index}")


    # ============================
    # LOCATING_DOORS
    # ============================

    def locate_door_step(self):
        """We hit a blocked cell; decide if this is a door worth opening."""
        if self.current_blocked_cell is None:
            self.get_logger().info(
                "LOCATING_DOORS: no blocked cell stored → going to UPDATING_MAP."
            )
            self.state = self.UPDATING_MAP
            return

        if not self.detected_doors:
            self.get_logger().info(
                "LOCATING_DOORS: blocked but no known doors → UPDATING_MAP."
            )
            self.state = self.UPDATING_MAP
            return

        bx, by = self.current_blocked_cell

        # choose the closest detected door to the blocked cell
        best = min(
            self.detected_doors,
            key=lambda d: math.hypot(d[0] - bx, d[1] - by),
        )

        self.current_door_cell = best
        self.get_logger().info(
            f"LOCATING_DOORS: blocked at {self.current_blocked_cell}, "
            f"choosing door at {self.current_door_cell} → OPENING_DOORS."
        )
        self.state = self.OPENING_DOORS

    # ============================
    # OPENING_DOORS
    # ============================

    def open_door_step(self):
        """Move to the chosen door and attempt to open it."""
        if self.current_door_cell is None:
            self.get_logger().info(
                "OPENING_DOORS: no door selected → UPDATING_MAP."
            )
            self.state = self.UPDATING_MAP
            return

        if self.keys_remaining <= 0:
            self.get_logger().info(
                "OPENING_DOORS: no keys remaining → UPDATING_MAP."
            )
            self.state = self.UPDATING_MAP
            return

        door_mx, door_my = self.current_door_cell
        wx, wy = self.map_to_world(door_mx, door_my)

        dist = math.hypot(self.drone_x - wx, self.drone_y - wy)

        # Move up to the door if we're not close enough yet
        if dist > 0.8:
            cmd = Vector3(x=float(wx), y=float(wy), z=0.0)
            self.cmd_pub.publish(cmd)
            return

        # Close enough to attempt the door
        if self.pending_door_future is not None:
            # we've already called the service; just wait for callback to update state
            return

        if (door_mx, door_my) in self.tried_door_cells:
            self.get_logger().info(
                "OPENING_DOORS: already tried this door → UPDATING_MAP."
            )
            self.state = self.UPDATING_MAP
            return

        self.get_logger().info(f"OPENING_DOORS: attempting door at {door_mx, door_my}")
        self.tried_door_cells.add((door_mx, door_my))
        self.try_open_door(wx, wy, door_mx, door_my)

    def try_open_door(self, wx, wy, mx, my):
        if not self.use_key_client.service_is_ready():
            self.get_logger().warn("use_key service not ready.")
            return

        req = UseKey.Request()
        req.door_loc = Point(x=float(wx), y=float(wy), z=0.0)

        future = self.use_key_client.call_async(req)
        self.pending_door_future = future

        def callback(fut, cell=(mx, my), self_ref=self):
            try:
                res = fut.result()
            except Exception as e:
                self_ref.get_logger().warn(f"use_key failed: {e}")
                self_ref.pending_door_future = None
                self_ref.state = self_ref.UPDATING_MAP
                return

            cmx, cmy = cell
            idx = self_ref.map_index(cmx, cmy)
            if idx < 0:
                self_ref.pending_door_future = None
                self_ref.state = self_ref.UPDATING_MAP
                return

            if res.success:
                # mark as open door now
                self_ref.grid[idx] = -2
                self_ref.get_logger().info(f"Door opened at {cell}")
            else:
                # mark as permanently closed / wall
                self_ref.grid[idx] = 100
                self_ref.get_logger().info(f"Door failed at {cell}")

            self_ref.pending_door_future = None
            # after opening/closing, replan
            self_ref.state = self_ref.UPDATING_MAP

        future.add_done_callback(callback)

    # ============================
    # UPDATING_MAP
    # ============================

    def update_map_and_replan(self):
        """Replan after a door interaction or blockage."""
        self.get_logger().info("UPDATING_MAP: replanning path after map change.")
        self.plan_path()
        self.current_blocked_cell = None
        self.current_door_cell = None

        if not self.current_path_world:
            # still no path; remain in UPDATING_MAP and just keep trying next ticks
            self.get_logger().warn(
                "UPDATING_MAP: no path found; will try again as map improves."
            )
            return

        # once we have a path, go back to moving
        self.state = self.MOVING_TO_WAYPOINT

    # ============================
    # Map + final path publishing
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
