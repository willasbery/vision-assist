from typing import ClassVar, Optional

from vision_assist.config import grid_size, penalty_colour_gradient


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
            if row_grids[mid].coords == current_coords:
                return mid
            elif row_grids[mid].coords.x < current_coords.x:
                left = mid + 1
            else:
                right = mid - 1
        return -1
    
    def calculate_row_penalty(self, current_grid: dict, grids: list[list[dict]]) -> float:
        """
        Calculate penalty for a grid based on its position within continuous segments.
        """
        row = current_grid.row
        row_grids = grids[row]
        current_coords = current_grid.coords
        
        if len(row_grids) == 1:
            return 0
        
        current_idx = self._find_grid_index(current_coords, row_grids)
        if current_idx == -1:
            return 0
        
        # Find segment boundaries
        x = current_coords.x
        furthest_left = current_idx
        furthest_right = current_idx
        
        # Find the furthest left grid in the segment
        while (furthest_left > 0 and 
               row_grids[furthest_left - 1].empty == False):
            furthest_left -= 1
            x -= grid_size
        
        # Find the furthest right grid in the segment
        x = current_coords.x
        while (furthest_right < len(row_grids) - 1 and 
               row_grids[furthest_right + 1].empty == False):
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


penalty_calculator = PenaltyCalculator()