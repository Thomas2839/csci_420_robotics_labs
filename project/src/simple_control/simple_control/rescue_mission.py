#!/usr/bin/env python3
"""Rescue mission node (checkpoint 1)

This node implements a simple occupancy grid generator for the project first
checkpoint. It subscribes to /uav/sensors/gps and /uav/sensors/lidar and
publishes an initial occupancy grid on /map (values initialized at 50).

Behavior implemented here is intentionally minimal to get the first
checkpoint working: we fuse the first few LIDAR scans into the map, marking
cells along each beam as free (decrease value) and marking the hit cell as
occupied (increase value). The node repeatedly publishes the current map so
the GUI can visualize it.
"""

from __future__ import annotations

import math
import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import LaserScan
from environment_controller.srv import UseKey

class RescueMission(Node):
	"""Simple ROS2 node producing an OccupancyGrid from LIDAR and GPS.

	This implementation aims to provide a reliable first checkpoint: the
	occupancy grid is created, initialized to 50 (unknown) and updated when
	LIDAR scans arrive. It publishes `/map` at a slow rate so the GUI and
	environment controller can see the result.
	"""

	def __init__(self):
		super().__init__('rescue_mission_node')

		# Map parameters
		default_w = 23
		default_h = 23
		self.declare_parameter('map_width', default_w)
		self.declare_parameter('map_height', default_h)

		self.map_width = int(self.get_parameter('map_width').get_parameter_value().integer_value)
		self.map_height = int(self.get_parameter('map_height').get_parameter_value().integer_value)

		# We use 1 meter per cell (resolution 1.0)
		self.resolution = 1.0

		# Map origin: bottom-left in world coordinates; drone starts at (0,0)
		# Shift the origin by +0.5 so cell centers align with integer
		# world coordinates. Without this shift, the center cell is offset by
		# -0.5 and the drone appears in the top-right corner of the center
		# cell (not in the exact middle). This change moves cell centers so
		# that the middle cell's center maps to world (0,0).
		self.origin_x = -self.map_width / 2.0 + 0.5
		self.origin_y = -self.map_height / 2.0 + 0.5

		# Occupancy grid stored as flat list in row-major (x,y) to match GUI
		self.grid = [50] * (self.map_width * self.map_height)

		# ROS publishers/subscribers
		self.map_pub = self.create_publisher(OccupancyGrid, '/map', 1)
		self.gps_sub = self.create_subscription(PoseStamped, '/uav/sensors/gps', self.gps_callback, 10)
		self.lidar_sub = self.create_subscription(LaserScan, '/uav/sensors/lidar', self.lidar_callback, 10)

		# Local state
		self.has_gps = False
		self.drone_x = 0.0
		self.drone_y = 0.0

		# Publish map periodically so GUI picks it up
		self.create_timer(0.5, self.publish_map)

		# Service client for opening doors (used for checkpoint demo)
		self.use_key_client = self.create_client(UseKey, 'use_key')
		self._tried_open_right = False
		# Small timer to attempt opening the door to the right of start
		self.create_timer(9, self._attempt_open_right)

		# Mark start cell as free (drone starts at 0,0)
		sx, sy = self.world_to_map(0.0, 0.0)
		self.set_cell(sx, sy, 0)

		self.get_logger().info(f'rescue_mission started: map {self.map_width}x{self.map_height}')

	# ----------------------- helper methods -----------------------
	def world_to_map(self, wx: float, wy: float) -> tuple[int, int]:
		"""Convert world coordinates to integer map indices.

		We want map indices to correspond to the cell whose center is
		nearest to the supplied world point. With the origin chosen above
		(cell centers on integer coordinates), this is equivalent to
		rounding to the nearest integer index. Using floor(x + 0.5)
		implements proper round-to-nearest behavior for both positive and
		negative coordinates without relying on Python's banker's rounding.
		"""
		mx = int(math.floor((wx - self.origin_x) + 0.5))
		my = int(math.floor((wy - self.origin_y) + 0.5))
		return mx, my

	def map_index(self, mx: int, my: int) -> int:
		if mx < 0 or my < 0 or mx >= self.map_width or my >= self.map_height:
			return -1
		return int(my + mx * self.map_height)

	def get_cell(self, mx: int, my: int) -> int | None:
		idx = self.map_index(mx, my)
		if idx < 0:
			return None
		return self.grid[idx]

	def set_cell(self, mx: int, my: int, value: int) -> bool:
		idx = self.map_index(mx, my)
		if idx < 0:
			return False
		# clamp the new value into [0,100]
		new_val = max(0, min(100, int(value)))

		# Preserve special sentinel values (doors/open/goal) represented by
		# negative numbers; we don't overwrite these here.
		cur = self.grid[idx]
		if cur < 0:
			# leave sentinel values alone
			return False

		# Hard sticky rule: if a cell has reached 0 (fully confident free),
		# do not let it be raised again by small/ambiguous obstacle evidence.
		# This prevents the 'white -> grey' visual fade you observed where
		# free cells become slightly uncertain again by small updates.
		#
		# Note (softer rule idea): instead of the hard protection above we
		# could implement a confirmation policy whereby a cell that becomes
		# free is marked as "confirmed-free" only after N independent
		# observations, or track a 'log-odds' fused value with different
		# thresholds and only accept an increase when log-odds push the
		# cell strongly back toward occupied. That approach is more robust
		# but requires maintaining additional state (e.g., counts or
		# log-odds) which is out-of-scope for this checkpoint.
		if cur == 0 and new_val > 0:
			# keep cell at 0 (confirmed free)
			return True

		# store the new value
		self.grid[idx] = new_val
		return True

	# ----------------------- callbacks -----------------------
	def gps_callback(self, msg: PoseStamped):
		self.drone_x = float(msg.pose.position.x)
		self.drone_y = float(msg.pose.position.y)
		self.has_gps = True

	def lidar_callback(self, msg: LaserScan):
		# Ignore scans until we have GPS (we need the origin)
		if not self.has_gps:
			return

		# Each LaserScan contains many range measurements in polar coordinates.
		# `angle_min` is the starting angle for the first beam and
		# `angle_increment` is the step between consecutive beams. We iterate
		# over each range reading and compute a world-space ray from the drone
		# origin in that direction.
		angle = msg.angle_min
		for r in msg.ranges:
			# Skip invalid negative values
			if math.isfinite(r) and r <= 0:
				angle += msg.angle_increment
				continue

			# Range semantics:
			# - If a reading `r` is finite and > 0, it is the distance to the
			#   closest obstacle along that beam.
			# - If `r` is infinite (the lidar saw no hit within range) we
			#   consider the beam to travel to the sensor's `range_max`.
			# - Negative or zero values are invalid and are skipped above.
			# We use math.isfinite to detect actual numeric hits vs. 'inf'. For
			# infinite beams, we still trace out to `range_max` so we can mark
			# free space along the whole beam.
			max_range = msg.range_max if math.isfinite(msg.range_max) else 5.0
			dist = r if math.isfinite(r) else max_range

			# We trace the beam in short increments from the drone position up
			# to (but not including) the obstacle location. Each visited cell is
			# considered evidence that the cell is free (we decrease its
			# occupancy value). The step length is chosen to be smaller than the
			# map resolution so we won't accidentally skip a cell.
			#
			# Choosing step size:
			# - Using a fraction of the resolution guarantees we sample inside
			#   every covered cell, even when the beam crosses cell boundaries
			#   diagonally. A small lower bound (0.1m) keeps the sampling
			#   stable even for high resolution maps.
			step = max(0.1, self.resolution / 4.0)
			t = 0.0
			# `last_mx_my` remembers the last map cell we updated while tracing
			# this beam. Because `step` is often much smaller than the cell
			# size, we will visit the same cell many times while stepping through
			# the beam. If we incrementally decrease the occupancy for every
			# step inside the same cell, values will drift continuously with
			# each scan (this produces the visual 'white -> grey' fading the
			# GUI user reported). By only updating each visited cell once per
			# beam (the first time we cross it), we avoid artificially
			# strengthening the free-space evidence for that cell from a single
			# beam.
			last_mx_my = None
			while t < dist:
				wx = self.drone_x + t * math.cos(angle)
				wy = self.drone_y + t * math.sin(angle)
				mx, my = self.world_to_map(wx, wy)
				if (mx, my) != last_mx_my:
					# mark this cell as more likely free (we saw unblocked space
				# here along the beam). This decreases the occupancy confidence
				# by a fixed amount. We skip special sentinel values that mark
				# doors (-1: closed, -2: opened) or the goal (-3), because those
				# should be preserved and not treated as normal cells.
					cur = self.get_cell(mx, my)
					if cur is not None and cur != -1 and cur != -2 and cur != -3:
						# Here we reduce by 15 — this is a heuristic value tuned to make
					# a single beam provide useful evidence without overwriting a
					# long history of observations. Future scans will continue to
					# push cells further toward 0 (free) if consistently seen as
					# free.
						self.set_cell(mx, my, (cur - 15))
					last_mx_my = (mx, my)
				t += step

			# If this beam actually hit an obstacle (a finite range) then we
			# mark the cell where the hit occurred as more likely occupied.
			# This increases the occupancy score for the hit cell by a larger
			# amount so a single hit moves the cell closer to fully occupied
			# (100). Note: doors and objects read by the LIDAR will be marked
			# as occupied; door-specific handling comes later in the project.
			if math.isfinite(r) and dist > 0:
				hx = self.drone_x + dist * math.cos(angle)
				hy = self.drone_y + dist * math.sin(angle)
				hm_x, hm_y = self.world_to_map(hx, hy)
				cur = self.get_cell(hm_x, hm_y)
				if cur is not None:
					# increase occupancy confidence by 30 — this is intentionally
					# larger than the decrement used for free cells so obstacle
					# evidence quickly dominates ambiguous cells.
					# The particular increments/decrements (30/15) are tunable
					# heuristics for this checkpoint to make the grid converge
					# quickly from a few scans. Later steps in the project should
					# refine these via probabilistic fusion (e.g. log-odds) or
					# noise-models for doors.
					self.set_cell(hm_x, hm_y, (cur + 30))

			angle += msg.angle_increment

	# ----------------------- publishing -----------------------
	def publish_map(self):
		msg = OccupancyGrid()
		msg.header.stamp = self.get_clock().now().to_msg()
		msg.header.frame_id = 'map'

		info = MapMetaData()
		info.resolution = float(self.resolution)
		info.width = int(self.map_width)
		info.height = int(self.map_height)
		info.origin.position.x = float(self.origin_x)
		info.origin.position.y = float(self.origin_y)
		info.origin.position.z = 0.0
		msg.info = info

		# copy our internal grid
		msg.data = [int(g) for g in self.grid]

		self.map_pub.publish(msg)

	def _attempt_open_right(self):
		"""Try to open the door immediately to the right of start (1,0).

		This method runs from a short timer at startup. It only tries once
		and requests the `use_key` service with the world point (1.0, 0.0).
		"""
		if self._tried_open_right:
			return
		self._tried_open_right = True

		# Ensure service exists (short timeout) — we don't block startup long
		if not self.use_key_client.wait_for_service(timeout_sec=1.0):
			self.get_logger().warning('use_key service not available (open-right)')
			return

		req = UseKey.Request()
		req.door_loc = Point()
		req.door_loc.x = 1.0
		req.door_loc.y = 0.0
		req.door_loc.z = 0.0

		future = self.use_key_client.call_async(req)
		future.add_done_callback(self._open_right_response)

	def _open_right_response(self, fut):
		try:
			res = fut.result()
		except Exception as e:
			self.get_logger().warning(f'use_key call for right door failed: {e}')
			return

		if getattr(res, 'success', False):
			# mark the right-hand cell as opened door sentinel (-2)
			mx, my = self.world_to_map(1.0, 0.0)
			idx = self.map_index(mx, my)
			if idx >= 0:
				self.grid[idx] = -2
			self.get_logger().info('Right-side door opened successfully')
		else:
			self.get_logger().info('Right-side door use_key returned false')


def main(args=None):
	rclpy.init(args=args)
	try:
		node = RescueMission()
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		rclpy.try_shutdown()


if __name__ == '__main__':
	main()

