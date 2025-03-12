import numpy as np
from heapq import heappop, heappush
from typing import ClassVar, Optional

from models import Grid

class PathFinder:
    """
    PathFinder handles pathfinding operations using the A* algorithm,
    penalising paths with sharp angles across multiple grids for smoother paths.
    """
    _instance: ClassVar[Optional['PathFinder']] = None
    _initialized: bool = False

    def __new__(cls):
        """Ensure only one instance of PathFinder exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the pathfinder only once."""
        if not self._initialized:
            self._initialized = True
            self._grid_lookup: dict[tuple[int, int], Grid] = {}
            self._open_set: list[tuple[float, tuple[int, int]]] = []
            self._closed_set: set[tuple[int, int]] = set()
            self._came_from: dict[tuple[int, int], tuple[int, int]] = {}
            self._g_score: dict[tuple[int, int], float] = {}
            self._f_score: dict[tuple[int, int], float] = {}

    def _reset_path_state(self) -> None:
        """Reset the internal state for a new pathfinding operation."""
        self._open_set.clear()
        self._closed_set.clear()
        self._came_from.clear()
        self._g_score.clear()
        self._f_score.clear()

    def _heuristic(self, grid1: dict, grid2: dict) -> float:
        """
        Calculate the Manhattan distance between two grids.
        """
        return abs(grid1.coords.x - grid2.coords.x) + \
               abs(grid1.coords.y - grid2.coords.y)

    def _angle_between_grids(self, path: list[tuple[int, int]], segment_size: int) -> float:
        """
        Calculate the angle changes over a sliding window of the path, considering
        both previous and upcoming segments for smoother transitions.

        Args:
            path: List of grid coordinates (x, y) representing the path so far.
            segment_size: Number of grids to consider for angle calculation.

        Returns:
            The maximum angle change across the analyzed segments.
        """
        if len(path) < segment_size:
            return 0

        np_path = np.array(path)
        angles = []
        # Use a sliding window approach to analyze segments
        half = segment_size // 2
        
        for i in range(half, len(path) - half - 1):
            # Get points before and after the current point
            vec1 = np_path[i] - np_path[i - half]
            vec2 = np_path[i + half] - np_path[i]
            
            # Calculate direction vectors for both segments
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0: continue

            # Calculate angle between vectors
            dot = np.dot(vec1, vec2)
            angle = np.degrees(np.arccos(np.clip(dot / (norm1 * norm2), -1.0, 1.0)))
            angles.append(angle)

        return max(angles) if angles else 0

    def _reconstruct_path(self, end_coords: tuple[int, int], start_grid: dict) -> tuple[list[Grid], float]:
        """
        Reconstruct the path from the came_from dictionary.
        """
        path = []
        current = end_coords
        total_cost = self._g_score[current]

        while current in self._came_from:
            path.append(self._grid_lookup[current])
            current = self._came_from[current]

        path.append(start_grid)
        path.reverse()
        return path, total_cost

    def find_path(
        self,
        graph: dict,
        start_grid: Grid,
        end_grid: Grid,
        grid_lookup: dict[tuple[int, int], Grid]
    ) -> tuple[list[Grid], float]:
        """
        Find the optimal path between start and end grids using A* algorithm,
        penalising sharp angle changes for smoother paths.
        """
        self._reset_path_state()
        self._grid_lookup = grid_lookup

        start_coords = (start_grid.coords.x, start_grid.coords.y)
        end_coords = (end_grid.coords.x, end_grid.coords.y)

        # Initialize scores for start position
        self._g_score[start_coords] = 0
        self._f_score[start_coords] = self._heuristic(start_grid, end_grid)
        heappush(self._open_set, (self._f_score[start_coords], start_coords))

        while self._open_set:
            current_coords = heappop(self._open_set)[1]

            if current_coords == end_coords:
                return self._reconstruct_path(end_coords, start_grid)

            self._closed_set.add(current_coords)

            # Process neighbours
            for neighbour_coords, distance in graph[current_coords]:
                if neighbour_coords in self._closed_set:
                    continue

                neighbour_grid = self._grid_lookup[neighbour_coords]

                # Retrieve the path so far
                path_so_far = [current_coords]
                previous = current_coords
                while previous in self._came_from:
                    previous = self._came_from[previous]
                    path_so_far.append(previous)
                path_so_far.reverse()

                # Calculate angle penalty based on larger segments
                segment_size = 7  # Consider 7 points: 3 before, current, and 3 after
                avg_angle_change = self._angle_between_grids(path_so_far + [neighbour_coords], segment_size)
                # More gradual penalty for angles
                angle_penalty = 0 if avg_angle_change <= 30 else (avg_angle_change / 90) ** 1.5

                # Adjust penalty multiplier to be more lenient
                penalty_multiplier = 1 + (0.5 * (neighbour_grid.penalty or 0)) + angle_penalty * 1.5
                # penalty_multiplier = 1 + (0.5 * (neighbour_grid.penalty or 0))
                tentative_g_score = self._g_score[current_coords] + (distance * penalty_multiplier)

                if (neighbour_coords not in self._g_score or
                        tentative_g_score < self._g_score[neighbour_coords]):
                    self._came_from[neighbour_coords] = current_coords
                    self._g_score[neighbour_coords] = tentative_g_score
                    self._f_score[neighbour_coords] = tentative_g_score + \
                                                      self._heuristic(neighbour_grid, end_grid)

                    if not any(coords == neighbour_coords for _, coords in self._open_set):
                        heappush(self._open_set,
                                 (self._f_score[neighbour_coords], neighbour_coords))

        return [], float('inf')


path_finder = PathFinder()