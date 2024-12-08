from typing import ClassVar, Optional

from config import grid_size, penalty_colour_gradient
from models import Coordinate, Grid



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
    
    def _calculate_segment_penalty(
        self, grid: Grid, grid_lookup: dict[tuple[int, int], Grid], direction: str
    ) -> float:
        """
        Calculate penalty for a grid within a segment (row or column).
        """
        current_coords = grid.coords
        x, y = current_coords.x, current_coords.y
        furthest_left, furthest_right = current_coords, current_coords

        # Traverse in the left/up direction
        while True:
            next_coords = (
                (x - grid_size, y) if direction == "row" else (x, y - grid_size)
            )
            if next_coords not in grid_lookup or grid_lookup[next_coords].empty:
                break
            furthest_left = Coordinate(x=next_coords[0], y=next_coords[1])
            x, y = next_coords

        # Reset x, y to the current coordinates
        x, y = current_coords.x, current_coords.y

        # Traverse in the right/down direction
        while True:
            next_coords = (
                (x + grid_size, y) if direction == "row" else (x, y + grid_size)
            )
            if next_coords not in grid_lookup or grid_lookup[next_coords].empty:
                break
            furthest_right = Coordinate(x=next_coords[0], y=next_coords[1])
            x, y = next_coords

        segment_width = (
            (furthest_right.x - furthest_left.x) // grid_size
            if direction == "row"
            else (furthest_right.y - furthest_left.y) // grid_size
        )
        if segment_width == 0:
            return 0

        position_ratio = (
            (current_coords.x - furthest_left.x) / (furthest_right.x - furthest_left.x)
            if direction == "row"
            else (current_coords.y - furthest_left.y)
            / (furthest_right.y - furthest_left.y)
        )
        return 2 * abs(position_ratio - 0.5)
    
    def calculate_penalty(self, grid: Grid, grid_lookup: dict[tuple[int, int], Grid]) -> float:
        """
        Calculate penalty for a grid based on its position within continuous segments
        in both its row and column using grid lookup.
        """
        if grid.empty:
            return 0
        
        # if grid.artificial:
        #     return 0.1

        # Calculate row and column penalties
        row_penalty = self._calculate_segment_penalty(grid, grid_lookup, "row")
        col_penalty = self._calculate_segment_penalty(grid, grid_lookup, "col")
        
        if row_penalty > 0.99 or col_penalty > 0.99:
            return 1

        total_penalty = row_penalty + col_penalty
        if total_penalty == 0:
            return 0

        # Calculate dominance factor
        dominance_factor = abs(row_penalty - col_penalty) / total_penalty
        row_weight = 0.5 + (0.25 * dominance_factor if row_penalty > col_penalty else -0.25 * dominance_factor)
        col_weight = 1 - row_weight

        # Weighted average
        penalty = (row_penalty * row_weight) + (col_penalty * col_weight)

        return penalty

    def get_penalty_colour(self, penalty: float) -> tuple[int, int, int]:
        """
        Get the colour based on the penalty value.
        """
        closest_key = min(
            penalty_colour_gradient.keys(),
            key=lambda x: abs(x - penalty),
        )
        return penalty_colour_gradient[closest_key]


penalty_calculator = PenaltyCalculator()
