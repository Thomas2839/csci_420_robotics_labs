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

        map_data: 2D numpy array with semantics above.
        drone_position, goal_position: [i, j] integer indices.
        """
        self.validate_positions(map_data, drone_position, goal_position)

        # Inflate obstacles by safety distance
        expanded = self.expand_obstacles(map_data, self.safe_distance)

        goal_n = Node(goal_position)

        frontier = PriorityQueue()
        closed = []
        counter = itertools.count()

        start = Node(drone_position)
        start.heuristic = self.movement_cost(start, goal_n)
        frontier.put((start.cost + start.heuristic, next(counter), start))

        while not frontier.empty():
            _, _, cur = frontier.get()

            # -------------------------------
            # Goal Test
            # -------------------------------
            if tuple(cur.pos) == tuple(goal_position):
                return self.reconstruct_path(cur)

            # -------------------------------
            # Expand neighbors
            # -------------------------------
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

                # If neighbor not present in either list
                if idx_f == -1 and idx_c == -1:
                    nbr.cost = new_cost
                    nbr.heuristic = self.movement_cost(nbr, goal_n)
                    nbr.parent = cur
                    frontier.put((nbr.cost + nbr.heuristic, next(counter), nbr))

            # Add to closed list
            if self.insidelist(cur, closed) == -1:
                closed.append(cur)

        # No path
        return None

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

                # ============================
                # WALKABLE LOGIC
                # ----------------------------
                # Free / likely free
                if 0 <= cell < 70:
                    neighbors.append(Node([ni, nj]))
                    continue

                # Open door (passable)
                if cell == -2:
                    neighbors.append(Node([ni, nj]))
                    continue

                # Goal (passable)
                if cell == -3:
                    neighbors.append(Node([ni, nj]))
                    continue

                # Everything else is blocked:
                #  -1 closed door
                #  70–100 obstacles
                # ============================

        return neighbors

    # ==================================================
    # Validation
    # ==================================================
    def validate_positions(self, map_data, start, goal):
        """Ensure positions are within bounds. Does NOT require them to be free."""
        assert len(map_data.shape) == 2
        assert len(start) == 2
        assert len(goal) == 2
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
        """
        Inflate *hard* obstacles (>=70) by 'distance' cells in all directions.
        Do NOT overwrite negative sentinel values (-1, -2, -3).
        """
        new_map = copy.deepcopy(map_data)

        for i in range(map_data.shape[0]):
            for j in range(map_data.shape[1]):
                # Only inflate hard obstacles, not doors/goals
                if map_data[i, j] >= 70:
                    for di in range(-distance, distance + 1):
                        for dj in range(-distance, distance + 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < new_map.shape[0] and 0 <= nj < new_map.shape[1]:
                                # Don't overwrite sentinel values
                                if new_map[ni, nj] >= 0:
                                    new_map[ni, nj] = max(new_map[ni, nj], 70)

        return new_map
