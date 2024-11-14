from heapq import heappop, heappush
from typing import ClassVar, Optional

from vision_assist.models import Grid


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
            self._grid_lookup: dict[tuple[int, int], Grid] = {}
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
        return abs(grid1.coords.x - grid2.coords.x) + \
               abs(grid1.coords.y - grid2.coords.y)
    
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
        start_grid: Grid,
        end_grid: Grid,
        grid_lookup: dict[tuple[int, int], Grid]
    ) -> tuple[list[dict], float]:
        """
        Find the optimal path between start and end grids using A* algorithm.
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
            
            # Process neighbors
            for neighbor_coords, distance in graph[current_coords]:
                if neighbor_coords in self._closed_set:
                    continue
                
                neighbor_grid = self._grid_lookup[neighbor_coords]
                penalty_multiplier = 1 + (neighbor_grid.penalty or 0)
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



path_finder = PathFinder()