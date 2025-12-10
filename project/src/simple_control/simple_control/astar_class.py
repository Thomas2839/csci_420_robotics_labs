from queue import PriorityQueue
import copy
import math
import itertools
from numbers import Integral
import numpy as np

# ============================
# Grid Value Semantics
# ----------------------------
#   0–69   : free / likely free
#   70–100 : obstacle / likely obstacle
#   -1     : closed door (blocked)
#   -2     : open door (passable)
#   -3     : goal (passable)
# ============================

class Node:
    def __init__(self, position):
        self.pos = position        # [i, j] map coordinates
        self.cost = 0.0
        self.heuristic = 0.0
        self.parent = None


class AStarPlanner:
    def __init__(self, safe_distance=1):
        self.safe_distance = safe_distance

    # ==================================================
    # Public API: plan()
    # ==================================================
    def plan(self, map_data, drone_position, goal_position):
        """
        Plan a path from start to goal on a probabilistic occupancy grid.

        map_data: 2D numpy array
        drone_position, goal_position: [i, j]
        """
        self.validate_positions(map_data, drone_position, goal_position)

        # ------------------------------------------------------
        # DOOR-AWARE GOAL REWRITE (only internal to A*)
        # If the goal is a DOOR (-1), route to best adjacent tile
        # ------------------------------------------------------
        gi, gj = goal_position
        if map_data[gi, gj] == -1:
            adj = []
            candidates = [
                (gi - 1, gj),  # up
                (gi + 1, gj),  # down
                (gi, gj - 1),  # left
                (gi, gj + 1),  # right
            ]

            for ci, cj in candidates:
                if 0 <= ci < map_data.shape[0] and 0 <= cj < map_data.shape[1]:
                    val = map_data[ci, cj]
                    # Walkable types
                    if (0 <= val < 70) or val in (-2, -3):
                        adj.append([ci, cj])

            if not adj:
                return None  # cannot reach any adjacent tile

            # Choose adjacent tile closest to the drone
            best_adj = min(
                adj,
                key=lambda p: math.hypot(
                    drone_position[0] - p[0], drone_position[1] - p[1]
                ),
            )

            goal_position = best_adj

        # Inflate obstacles by safety distance
        expanded = self.expand_obstacles(map_data, self.safe_distance)

        goal_n = Node(goal_position)
        #front
        frontier = PriorityQueue()
        closed = []
        counter = itertools.count()

        start = Node(drone_position)
        start.heuristic = self.movement_cost(start, goal_n)
        frontier.put((start.cost + start.heuristic, next(counter), start))

        # ==================================================
        # Main A* loop
        # ==================================================
        while not frontier.empty():
            _, _, cur = frontier.get()

            # Goal test
            if tuple(cur.pos) == tuple(goal_position):
                return self.reconstruct_path(cur)

            # Expand neighbors
            for nbr in self.get_neighbors(cur, expanded):
                new_cost = cur.cost + self.movement_cost(cur, nbr)

                frontier_nodes = self.priorityQueueToList(frontier)
                idx_f = self.insidelist(nbr, frontier_nodes)
                idx_c = self.insidelist(nbr, closed)

                # Frontier replacement logic
                if idx_f != -1 and new_cost < frontier.queue[idx_f][2].cost:
                    frontier.queue.pop(idx_f)

                # Closed replacement logic
                if idx_c != -1 and new_cost < closed[idx_c].cost:
                    closed.pop(idx_c)

                # New node
                if idx_f == -1 and idx_c == -1:
                    nbr.cost = new_cost
                    nbr.heuristic = self.movement_cost(nbr, goal_n)
                    nbr.parent = cur
                    frontier.put((nbr.cost + nbr.heuristic, next(counter), nbr))

            # Add to closed list
            if self.insidelist(cur, closed) == -1:
                closed.append(cur)

        return None  # No path found

    # ==================================================
    # Reconstruct final path
    # ==================================================
    def reconstruct_path(self, node):
        path = []
        while node is not None:
            path.append(node.pos)
            node = node.parent
        return list(reversed(path))

    # ==================================================
    # Cost function (Euclidean distance)
    # ==================================================
    def movement_cost(self, n1, n2):
        return math.hypot(n1.pos[0] - n2.pos[0], n1.pos[1] - n2.pos[1])

    # ==================================================
    # Node membership helpers
    # ==================================================
    def insidelist(self, node_in, list_in):
        for i, v in enumerate(list_in):
            if tuple(v.pos) == tuple(node_in.pos):
                return i
        return -1

    def priorityQueueToList(self, queue_in):
        return [item[2] for item in queue_in.queue]

    # ==================================================
    # Neighbor generation with semantics
    # ==================================================
    def get_neighbors(self, node_in, map_in):
        neighbors = []
        i, j = node_in.pos

        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj

                # Bounds check
                if not (0 <= ni < map_in.shape[0] and 0 <= nj < map_in.shape[1]):
                    continue

                cell = map_in[ni, nj]

                # Walkable classes:
                if 0 <= cell < 70:       # free
                    neighbors.append(Node([ni, nj]))
                elif cell in (-2, -3):   # open door or goal
                    neighbors.append(Node([ni, nj]))
                # closed door (-1) and obstacles (70–100) are blocked

        return neighbors

    # ==================================================
    # Validation
    # ==================================================
    def validate_positions(self, map_data, start, goal):
        assert len(map_data.shape) == 2
        assert len(start) == 2 and len(goal) == 2
        for x in start + goal:
            assert isinstance(x, Integral)
        assert 0 <= start[0] < map_data.shape[0]
        assert 0 <= start[1] < map_data.shape[1]
        assert 0 <= goal[0] < map_data.shape[0]
        assert 0 <= goal[1] < map_data.shape[1]

    # ==================================================
    # Obstacle inflation
    # ==================================================
    def expand_obstacles(self, map_data, distance):
        new_map = copy.deepcopy(map_data)

        for i in range(map_data.shape[0]):
            for j in range(map_data.shape[1]):
                if map_data[i, j] >= 70:   # hard obstacle
                    for di in range(-distance, distance + 1):
                        for dj in range(-distance, distance + 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < new_map.shape[0] and 0 <= nj < new_map.shape[1]:
                                if new_map[ni, nj] >= 0:  # don't overwrite sentinel values
                                    new_map[ni, nj] = max(new_map[ni, nj], 70)

        return new_map
