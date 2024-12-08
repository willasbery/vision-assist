from pydantic import BaseModel
from typing import Literal


class Coordinate(BaseModel):
    x: int
    y: int
    
    def to_tuple(self) -> tuple[int, int]:
        return (self.x, self.y)

class Grid(BaseModel):
    coords: Coordinate
    centre: Coordinate
    penalty: float | None # is None when the grid is empty
    row: int
    col: int
    empty: bool
    
class Corner(BaseModel):
    direction: Literal["left", "right"]
    sharpness: Literal["sharp", "sweeping"]
    start: Coordinate
    end: Coordinate
    angle_change: float
    
class Obstacle(BaseModel):
    type: Literal["left", "right", "forward_instruct_left", "forward_instruct_right"]
    bbox: tuple[int, int, int, int] # x, y, w, h
    distance: float
    angle: float
    confidence: float
    
    @property
    def centre(self) -> tuple[int, int]:
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)