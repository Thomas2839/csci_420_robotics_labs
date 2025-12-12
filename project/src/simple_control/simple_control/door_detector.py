#!/usr/bin/env python3
"""Simple door detection helper.

Collects spaced LiDAR samples per beam and flags beams whose
standard deviation is a statistical outlier compared to the rest.

This is intentionally lightweight and agnostic of map frames — it
only operates on scan ranges and returns beam indices that are
standouts so the caller can convert to world/map coordinates.
"""
from __future__ import annotations

import math
from collections import deque
import statistics
import time
from typing import List, Tuple


class DoorDetector:
    def __init__(self, buffer_size: int = 6, sample_interval: float = 1.0, std_multiplier: float = 3.0):
        # number of historical samples to keep per beam
        self.buffer_size = int(buffer_size)
        # minimum seconds between stored samples (spacing to avoid jitter)
        self.sample_interval = float(sample_interval)
        # how many population-stddevs above the mean a beam must be to be flagged
        self.std_multiplier = float(std_multiplier)

        self._last_sample_time = 0.0
        self._beam_buffers: List[deque] = []
        self._angle_count = 0
        self._max_range = 5.0
        # track recently reported door candidates to avoid spam
        self._recent_candidates: dict = {}  # (mx, my) -> time_last_reported

    def sample_scan(self, scan) -> bool:
        """Store one sample from a LaserScan if enough time has elapsed.

        Returns True when a sample was stored (so detection can be run).
        """
        now = time.time()
        if now - self._last_sample_time < self.sample_interval:
            return False
        self._last_sample_time = now

        ranges = list(scan.ranges)
        if len(ranges) == 0:
            return False

        if self._angle_count != len(ranges):
            # reinitialize buffers when scan layout changes
            self._angle_count = len(ranges)
            self._beam_buffers = [deque(maxlen=self.buffer_size) for _ in range(self._angle_count)]
            if math.isfinite(scan.range_max):
                self._max_range = scan.range_max

        for i, r in enumerate(ranges):
            val = r if math.isfinite(r) else self._max_range
            self._beam_buffers[i].append(float(val))

        return True

    def detect_standouts(self) -> List[Tuple[int, float]]:
        """Return list of (beam_index, stddev) for beams that are outliers.

        Uses population stddev across buffered samples for each beam and
        flags beams whose stddev is > mean + std_multiplier * stdev_of_stds.
        """
        if not self._beam_buffers or any(len(b) < 2 for b in self._beam_buffers):
            return []

        stds = []
        for buf in self._beam_buffers:
            try:
                s = statistics.pstdev(buf)
            except Exception:
                s = 0.0
            stds.append(s)

        mean_std = statistics.mean(stds)
        # use population stdev of the std list if possible
        stdev_of_stds = statistics.pstdev(stds) if len(stds) > 1 else 0.0

        candidates: List[Tuple[int, float]] = []
        threshold = mean_std + self.std_multiplier * stdev_of_stds

        for i, s in enumerate(stds):
            # require s to be noticeably larger than baseline
            if s > threshold and s > 1e-3:
                candidates.append((i, s))

        return candidates

    def latest_range_for(self, beam_index: int) -> float:
        if 0 <= beam_index < len(self._beam_buffers) and len(self._beam_buffers[beam_index]) > 0:
            return float(self._beam_buffers[beam_index][-1])
        return float(self._max_range)

    def should_report_candidate(self, mx: int, my: int, report_cooldown: float = 5.0) -> bool:
        """Check if a candidate cell should be reported (with cooldown to avoid spam)."""
        now = time.time()
        key = (int(mx), int(my))
        if key in self._recent_candidates:
            if now - self._recent_candidates[key] < report_cooldown:
                return False
        self._recent_candidates[key] = now
        return True
