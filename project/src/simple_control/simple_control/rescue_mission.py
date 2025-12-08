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
import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import PoseStamped, Point, Vector3, PointStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32, Int32MultiArray

from environment_controller.srv import UseKey

import numpy as np
import tf2_ros
from tf2_ros import TransformException
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point  # (import kept if you need it later)

from simple_control.astar_class import AStarPlanner


class RescueMission(Node):
	"""Combined mapping + planning node with a simple finite state machine."""

	# -------------------- Control States --------------------
	EXPLORING_WORLD = 0      # building map, waiting for goal
	UPDATING_MAP = 1         # (re)planning based on latest map
	MOVING_TO_WAYPOINT = 2   # following the current global path
	LOCATING_DOORS = 3       # hit an obstacle → treat as potential door
	OPENING_DOORS = 4        # actively calling use_key on a door
	AT_GOAL = 5              # mission complete

	def __init__(self):
		super().__init__('rescue_mission_node')

		# ============================
		# MAPPING SETUP
		# ============================

		default_w = 23
		default_h = 23
		self.declare_parameter('map_width', default_w)
		self.declare_parameter('map_height', default_h)

		self.map_width = int(self.get_parameter('map_width').value)
		self.map_height = int(self.get_parameter('map_height').value)

		# 1 meter per cell
		self.resolution = 1.0

		# Origin so that (0,0) is near the center of the map
		self.origin_x = -self.map_width / 2.0 + 0.5
		self.origin_y = -self.map_height / 2.0 + 0.5

		# Occupancy grid as flat list (row-major with custom indexing)
		# Semantics:
		#   0–69  : free / likely free
		#   70–100: obstacle / likely obstacle
		#   -1    : closed door (blocked)
		#   -2    : open door (passable)
		#   -3    : goal (passable)
		self.grid = [50] * (self.map_width * self.map_height)

		# TF buffer/listener
		self.tf_buffer = tf2_ros.Buffer()
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

		# Static /world -> /map transform broadcaster
		self.map_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
		self.publish_map_tf()

		# ============================
		# ROS pubs/subs
		# ============================

		# publishers
		self.map_pub = self.create_publisher(OccupancyGrid, '/map', 1)
		self.cmd_pub = self.create_publisher(Vector3, "/uav/input/position", 10)
		self.final_path_pub = self.create_publisher(Int32MultiArray, "/uav/final_path", 1)
		self.debug_path_pub = self.create_publisher(Int32MultiArray, "/uav/path", 1)

		# subscribers
		self.gps_sub = self.create_subscription(PoseStamped, '/uav/sensors/gps', self.gps_callback, 10)
		self.lidar_sub = self.create_subscription(LaserScan, '/uav/sensors/lidar', self.lidar_callback, 10)
		self.key_sub = self.create_subscription(Int32, "/keys_remaining", self.keys_cb, 10)
		self.cell_sub = self.create_subscription(Point, "/cell_tower/position", self.cell_tower_cb, 10)

		# door-opening service
		self.use_key_client = self.create_client(UseKey, 'use_key')

		# ============================
		# Internal state
		# ============================

		self.has_gps = False
		self.drone_x = 0.0
		self.drone_y = 0.0

		self.keys_remaining = 4
		self.has_goal = False
		self.goal_world = None        # (wx, wy)
		self.goal_cell = None         # (mx, my) in map indices

		# path following
		self.current_path_cells = []   # [(mx,my), ...]
		self.current_path_world = []   # [(wx,wy), ...]
		self.wp_index = 0
		self.tried_door_cells = set()  # cells we've already tried to open

		# A* planner instance
		self.astar_planner = AStarPlanner(safe_distance=1)

		# door currently being opened (wx, wy, mx, my)
		self.pending_door = None

		# finite state machine initial state
		self.state = self.EXPLORING_WORLD

		# timers
		self.create_timer(0.5, self.publish_map)
		self.create_timer(0.2, self.mission_step)

		# mark start cell as free
		sx, sy = self.world_to_map(0.0, 0.0)
		self.set_cell(sx, sy, 0)

		self.get_logger().info(
			f"Rescue mission initialized. Map {self.map_width}x{self.map_height}"
		)

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
		# keep sentinel values for doors/goal
		if cur < 0:
			return
		# don't re-occupy confirmed free cells
		if cur == 0 and val > 0:
			return

		self.grid[idx] = max(-3, min(100, int(val)))

	# ============================
	# Callbacks
	# ============================

	def gps_callback(self, msg: PoseStamped):
		self.drone_x = float(msg.pose.position.x)
		self.drone_y = float(msg.pose.position.y)
		self.has_gps = True

	def keys_cb(self, msg: Int32):
		self.keys_remaining = msg.data

	def cell_tower_cb(self, msg: Point):
		"""Convert dog's world position → map frame using TF."""
		if self.has_goal:
			return

		# Build a stamped point in the /world frame
		ps = PointStamped()
		ps.header.stamp = self.get_clock().now().to_msg()
		ps.header.frame_id = "world"
		ps.point = msg  # msg is a Point

		try:
			# Transform dog position into /map frame
			dog_in_map = self.tf_buffer.transform(ps, "map")
		except TransformException as e:
			self.get_logger().warn(f"TF transform failed: {e}")
			return

		gx = dog_in_map.point.x
		gy = dog_in_map.point.y

		self.goal_world = (gx, gy)

		mx, my = self.world_to_map(gx, gy)
		self.goal_cell = (mx, my)

		idx = self.map_index(mx, my)
		if idx >= 0:
			self.grid[idx] = -3  # goal sentinel

		self.has_goal = True

		self.get_logger().info(
			f"Dog TF-corrected: world→map gives map=({mx},{my})"
		)

	# ============================
	# LIDAR mapping
	# ============================

	def lidar_callback(self, msg: LaserScan):
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

			# free space along beam
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

			# obstacle hit
			if math.isfinite(r) and dist > 0:
				hx = self.drone_x + dist * math.cos(angle)
				hy = self.drone_y + dist * math.sin(angle)
				mx, my = self.world_to_map(hx, hy)
				cur = self.get_cell(mx, my)
				if cur is not None:
					self.set_cell(mx, my, cur + 30)

			angle += msg.angle_increment

	# ============================
	# A* interface
	# ============================

	def plan_path(self) -> bool:
		"""Call AStarPlanner using current probabilistic map + positions."""
		if not (self.has_gps and self.has_goal and self.goal_cell is not None):
			self.get_logger().warn("Cannot plan: missing GPS or goal.")
			return False

		start = self.world_to_map(self.drone_x, self.drone_y)
		goal = self.goal_cell
		sx, sy = start
		gx, gy = goal

		# reshape grid into 2D map with full semantics
		grid_np = np.array(self.grid, dtype=np.int16).reshape(
			(self.map_width, self.map_height)
		)
		map_for_astar = grid_np.copy()

		# ensure start and goal are marked passable for planning
		if 0 <= sx < self.map_width and 0 <= sy < self.map_height:
			map_for_astar[sx, sy] = 0
		if 0 <= gx < self.map_width and 0 <= gy < self.map_height:
			map_for_astar[gx, gy] = -3  # goal

		try:
			path_cells = self.astar_planner.plan(
				map_for_astar,
				[int(sx), int(sy)],
				[int(gx), int(gy)],
			)
		except AssertionError as e:
			self.get_logger().warn(f"A* validate_positions failed: {e}")
			return False

		if not path_cells:
			self.get_logger().warn("A* failed to find a path.")
			return False

		# Save path in both map and world coords
		self.current_path_cells = [(int(i), int(j)) for (i, j) in path_cells]
		self.current_path_world = [self.map_to_world(mx, my) for (mx, my) in self.current_path_cells]
		self.wp_index = 0
		self.tried_door_cells.clear()

		# publish debug path
		flat = []
		for mx, my in self.current_path_cells:
			flat.extend([mx, my])
		self.debug_path_pub.publish(Int32MultiArray(data=flat))

		self.get_logger().info(f"Planned path with {len(self.current_path_cells)} waypoints.")
		return True

	# ============================
	# Path following + door handling
	# ============================

	def follow_path(self):
		if not self.current_path_world:
			# nothing to follow → replan
			self.state = self.UPDATING_MAP
			return

		if self.wp_index >= len(self.current_path_world):
			self.get_logger().info("Reached the goal. Mission complete!")
			self.publish_final_path()
			self.state = self.AT_GOAL
			return

		wx, wy = self.current_path_world[self.wp_index]
		mx, my = self.world_to_map(wx, wy)
		cell = self.get_cell(mx, my)

		# ---------------------------
		# BLOCKED CELL HANDLING
		# ---------------------------
		# Blocked if:
		#   - outside map (cell is None)
		#   - obstacle confidence high (>= 70)
		#   - closed door sentinel (-1)
		if cell is None or cell >= 70 or cell == -1:
			# Only treat as DOOR if it's explicitly a closed-door sentinel (-1)
			if (cell == -1 and
				self.keys_remaining > 0 and
				(mx, my) not in self.tried_door_cells):

				dist = math.hypot(wx - self.drone_x, wy - self.drone_y)
				# only try to open if we're close to the door
				if dist <= 1.0:
					self.get_logger().info(f"Door candidate at ({mx},{my})")
					self.tried_door_cells.add((mx, my))
					self.pending_door = (wx, wy, mx, my)
					self.state = self.LOCATING_DOORS
					return

			# Otherwise it's just an obstacle or already sealed door: replan
			self.state = self.UPDATING_MAP
			return

		# ---------------------------
		# NORMAL MOTION
		# ---------------------------
		# Move TOWARD waypoint (unit direction)
		dx = wx - self.drone_x
		dy = wy - self.drone_y
		dist = math.hypot(dx, dy)
		if dist > 0:
			dx /= dist
			dy /= dist

		cmd = Vector3(x=dx, y=dy, z=0.0)
		self.cmd_pub.publish(cmd)

		# ---------------------------
		# WAYPOINT REACHED
		# ---------------------------
		if math.hypot(self.drone_x - wx, self.drone_y - wy) < 0.4:
			# If this waypoint is an OPEN DOOR, we assume we've just passed through it.
			# → Immediately "seal" it as a wall so future paths won't reconsider it.
			if cell == -2:
				idx = self.map_index(mx, my)
				if idx >= 0:
					self.grid[idx] = 100  # sealed door becomes hard wall
					self.get_logger().info(
						f"Sealed door at ({mx},{my}) after passing through."
					)

			self.wp_index += 1

	# ============================
	# Door opening
	# ============================

	def try_open_door(self, wx, wy, mx, my):
		if not self.use_key_client.service_is_ready():
			self.get_logger().warn("use_key service not ready.")
			return

		req = UseKey.Request()
		req.door_loc = Point(x=float(wx), y=float(wy), z=0.0)

		future = self.use_key_client.call_async(req)

		def callback(fut, cell=(mx, my)):
			try:
				res = fut.result()
			except Exception as e:
				self.get_logger().warn(f"use_key failed: {e}")
				return

			cmx, cmy = cell
			idx = self.map_index(cmx, cmy)
			if idx < 0:
				return

			if res.success:
				self.grid[idx] = -2   # opened door (passable for now)
				self.get_logger().info(f"Door opened at {cell}")
			else:
				self.grid[idx] = -1   # confirmed closed/blocked
				self.get_logger().info(f"Door failed at {cell}")

		future.add_done_callback(callback)

	# ============================
	# FSM step
	# ============================

	def mission_step(self):
		# light logging; comment out if too spammy
		self.get_logger().info(
			f"STATE={self.state}, GPS={self.has_gps}, goal={self.has_goal}, "
			f"pos=({self.drone_x:.2f},{self.drone_y:.2f})"
		)

		# No GPS → can't safely do anything
		if not self.has_gps:
			return

		if self.state == self.EXPLORING_WORLD:
			# building map + waiting for goal
			if self.has_goal:
				self.get_logger().info("Goal acquired → plan using current map.")
				self.state = self.UPDATING_MAP

		elif self.state == self.UPDATING_MAP:
			# (Re)run A* on the latest occupancy grid
			if self.plan_path():
				self.state = self.MOVING_TO_WAYPOINT

		elif self.state == self.MOVING_TO_WAYPOINT:
			self.follow_path()

		elif self.state == self.LOCATING_DOORS:
			# 'Locating' is trivial here: pending_door is already set
			if self.pending_door is not None:
				self.state = self.OPENING_DOORS
			else:
				# something went weird → just replan
				self.state = self.UPDATING_MAP

		elif self.state == self.OPENING_DOORS:
			if self.pending_door is not None:
				wx, wy, mx, my = self.pending_door
				self.try_open_door(wx, wy, mx, my)
				self.pending_door = None
			# after requesting door open, replan once the map updates
			self.state = self.UPDATING_MAP

		elif self.state == self.AT_GOAL:
			# Nothing else to do
			return

	# ============================
	# Publish map + final path
	# ============================

	def publish_map(self):
		msg = OccupancyGrid()
		msg.header.stamp = self.get_clock().now().to_msg()
		msg.header.frame_id = "map"

		info = MapMetaData()
		info.resolution = float(self.resolution)
		info.width = int(self.map_width)
		info.height = int(self.map_height)
		info.origin.position.x = float(self.origin_x)
		info.origin.position.y = float(self.origin_y)
		info.origin.position.z = 0.0
		msg.info = info

		msg.data = [int(v) for v in self.grid]
		self.map_pub.publish(msg)

	def publish_final_path(self):
		flat = []
		for mx, my in self.current_path_cells:
			flat.extend([mx, my])
		self.final_path_pub.publish(Int32MultiArray(data=flat))
		self.get_logger().info("Published final path.")

	def publish_map_tf(self):
		from geometry_msgs.msg import TransformStamped

		t = TransformStamped()
		t.header.stamp = self.get_clock().now().to_msg()
		t.header.frame_id = "world"
		t.child_frame_id = "map"

		# map origin in world coords
		t.transform.translation.x = self.origin_x
		t.transform.translation.y = self.origin_y
		t.transform.translation.z = 0.0
		t.transform.rotation.w = 1.0

		self.map_broadcaster.sendTransform(t)


def main(args=None):
	rclpy.init(args=args)
	node = RescueMission()
	rclpy.spin(node)
	rclpy.shutdown()


if __name__ == "__main__":
	main()
