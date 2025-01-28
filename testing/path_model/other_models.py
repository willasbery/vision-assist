from pydantic import BaseModel, computed_field
from typing import Literal

grid_size = 20


class Coordinate(BaseModel):
    x: int
    y: int
    
    @computed_field
    @property
    def midpoint(self) -> tuple[int, int]:
        return (self.x + (grid_size // 2), self.y + (grid_size // 2))
    
    def to_tuple(self) -> tuple[int, int]:
        return (self.x, self.y)

class Grid(BaseModel):
    coords: Coordinate
    centre: Coordinate
    penalty: float | None # is None when the grid is empty
    row: int
    col: int
    empty: bool
    artificial: bool
    
class Corner(BaseModel):
    direction: Literal["left", "right"]
    sharpness: Literal["sharp", "sweeping"]
    shape: Literal["inner", "outer", "optimal"]
    start: Coordinate
    end: Coordinate
    angle_change: float
    
    #test data attributes
    nearest_grid: Grid
    midpoint: Coordinate
    euclidean_distance: float
    