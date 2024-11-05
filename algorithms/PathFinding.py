import numpy as np
from heapq import heappop, heappush
from dataclasses import dataclass
from typing import ClassVar, Optional

from config import grid_size, penalty_colour_gradient


@dataclass
class GridCoordinates:
    """Represents grid coordinates in the path finding system."""
    x: int
    y: int
    
    def to_tuple(self) -> tuple[int, int]:
        """Convert coordinates to tuple format."""
        return (self.x, self.y)
    
    @classmethod
    def from_dict(cls, coords_dict: dict[str, int]) -> 'GridCoordinates':
        """Create GridCoordinates from a dictionary."""
        return cls(coords_dict['x'], coords_dict['y'])


class PathFinder:
    """
    PathFinder handles path finding operations using the A* algorithm.
    """
    _instance: ClassVar[Optional['PathFinder']] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Ensure only one instance of PathFinder exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the path finder only once."""
        if not self._initialized:
            self._initialized = True
            self._grid_lookup: dict[tuple[int, int], dict] = {}
            self._open_set: list[tuple[float, tuple[int, int]]] = []
            self._closed_set: set[tuple[int, int]] = set()
            self._came_from: dict[tuple[int, int], tuple[int, int]] = {}
            self._g_score: dict[tuple[int, int], float] = {}
            self._f_score: dict[tuple[int, int], float] = {}
    
    def _reset_path_state(self) -> None:
        """Reset the internal state for a new path finding operation."""
        self._open_set.clear()
        self._closed_set.clear()
        self._came_from.clear()
        self._g_score.clear()
        self._f_score.clear()
    
    def _heuristic(self, grid1: dict, grid2: dict) -> float:
        """
        Calculate the Manhattan distance between two grids.
        """
        return abs(grid1['coords']['x'] - grid2['coords']['x']) + \
               abs(grid1['coords']['y'] - grid2['coords']['y'])
    
    def _reconstruct_path(self, end_coords: tuple[int, int], start_grid: dict) -> tuple[list[dict], float]:
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
        start_grid: dict,
        end_grid: dict,
        grid_lookup: dict[tuple[int, int], dict]
    ) -> tuple[list[dict], float]:
        """
        Find the optimal path between start and end grids using A* algorithm.
        """
        self._reset_path_state()
        self._grid_lookup = grid_lookup
        
        start_coords = (start_grid['coords']['x'], start_grid['coords']['y'])
        end_coords = (end_grid['coords']['x'], end_grid['coords']['y'])
        
        # Initialize scores for start position
        self._g_score[start_coords] = 0
        self._f_score[start_coords] = self._heuristic(start_grid, end_grid)
        heappush(self._open_set, (self._f_score[start_coords], start_coords))
        
        while self._open_set:
            current_coords = heappop(self._open_set)[1]
            current_grid = self._grid_lookup[current_coords]
            
            if current_coords == end_coords:
                return self._reconstruct_path(end_coords, start_grid)
            
            self._closed_set.add(current_coords)
            
            # Process neighbors
            for neighbor_coords, distance in graph[current_coords]:
                if neighbor_coords in self._closed_set:
                    continue
                
                neighbor_grid = self._grid_lookup[neighbor_coords]
                penalty_multiplier = 1 + (neighbor_grid['penalty'] or 0)
                tentative_g_score = self._g_score[current_coords] + (distance * penalty_multiplier)
                
                if (neighbor_coords not in self._g_score or 
                    tentative_g_score < self._g_score[neighbor_coords]):
                    self._came_from[neighbor_coords] = current_coords
                    self._g_score[neighbor_coords] = tentative_g_score
                    self._f_score[neighbor_coords] = (tentative_g_score + 
                                                    self._heuristic(neighbor_grid, end_grid))
                    
                    if not any(coords == neighbor_coords for _, coords in self._open_set):
                        heappush(self._open_set, 
                               (self._f_score[neighbor_coords], neighbor_coords))
        
        return [], float('inf')


class PenaltyCalculator:
    """
    Handles the calculation and management of grid penalties.
    """
    _instance: ClassVar[Optional['PenaltyCalculator']] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Ensure only one instance of PenaltyCalculator exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the penalty calculator only once."""
        if not self._initialized:
            self._initialized = True
    
    def _find_grid_index(self, current_coords: dict, row_grids: list[dict]) -> int:
        """
        Find the index of a grid in a row using binary search.
        """
        left, right = 0, len(row_grids) - 1
        while left <= right:
            mid = (left + right) // 2
            if row_grids[mid]['coords'] == current_coords:
                return mid
            elif row_grids[mid]['coords']['x'] < current_coords['x']:
                left = mid + 1
            else:
                right = mid - 1
        return -1
    
    def calculate_row_penalty(self, current_grid: dict, grids: list[list[dict]]) -> float:
        """
        Calculate penalty for a grid based on its position within continuous segments.
        """
        row = current_grid['row']
        row_grids = grids[row]
        current_coords = current_grid['coords']
        
        if len(row_grids) == 1:
            return 0
        
        current_idx = self._find_grid_index(current_coords, row_grids)
        if current_idx == -1:
            return 0
        
        # Find segment boundaries
        x = current_coords['x']
        furthest_left = current_idx
        furthest_right = current_idx
        
        # Find the furthest left grid in the segment
        while (furthest_left > 0 and 
               row_grids[furthest_left - 1]['coords']['x'] == x - grid_size):
            furthest_left -= 1
            x -= grid_size
        
        # Find the furthest right grid in the segment
        x = current_coords['x']
        while (furthest_right < len(row_grids) - 1 and 
               row_grids[furthest_right + 1]['coords']['x'] == x + grid_size):
            furthest_right += 1
            x += grid_size
        
        segment_width = furthest_right - furthest_left
        if segment_width == 0:
            return 0
            
        position_ratio = (current_idx - furthest_left) / segment_width
        return 2 * abs(position_ratio - 0.5)
    
    def get_penalty_colour(self, penalty: float) -> tuple[int, int, int]:
        """
        Get the color based on the penalty value.
        """
        closest_key = min(penalty_colour_gradient.keys(), 
                         key=lambda x: abs(x - penalty))
        return penalty_colour_gradient[closest_key]


path_finder = PathFinder()
penalty_calculator = PenaltyCalculator()