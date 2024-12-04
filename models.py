from __future__ import annotations # for forward reference

import numpy as np
from pydantic import BaseModel, computed_field
from typing import Literal, Any


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
    
class Peak(BaseModel):
    centre: Coordinate
    left: Coordinate | None = None
    right: Coordinate | None = None
    orientation: Literal["left", "right", "up"] 
    
class ConvexityDefect(BaseModel):
    start: Coordinate
    end: Coordinate
    far: Coordinate
    depth: float
    
    @computed_field
    @property
    def angle_degrees(self) -> float:
        v1 = np.array(self.start.to_tuple()) - np.array(self.far.to_tuple())
        v2 = np.array(self.end.to_tuple()) - np.array(self.far.to_tuple())
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return np.degrees(angle) 
    
class Corner(BaseModel):
    type: Literal["left", "right"]
    sharpness: Literal["sharp", "sweeping"]
    confidence: float
    angle_change: float
    
class Obstacle(BaseModel):
    type: Literal["left", "right", "forward_instruct_left", "forward_instruct_right"]
    bbox: tuple[int, int, int, int] # x, y, w, h
    distance: float
    angle: float
    confidence: float
    
    @computed_field
    @property
    def centre(self) -> tuple[int, int]:
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)
    
class PathColours(BaseModel):
    close: tuple[int, int, int]
    mid: tuple[int, int, int]
    far: tuple[int, int, int]

class Path(BaseModel):
    """
    Represents a path through a grid with its properties and sections.
    Each path can have sub-sections which are also Path objects.
    """
    grids: list[Grid]
    total_cost: float
    path_type: Literal["path", "section"]
    sections: list[Path] | None = None  # a path section cannot have sections
    
    corners: list[Corner] | None = None  # a path section cannot have a corner 
    obstacles: list[Obstacle] | None = None  # a path section cannot have obstacles
    
    def model_post_init(self, __context: Any) -> None:
        if self.path_type == "path" and self.grids:  # only create sections for main path
            print(self.grids)
            self._calculate_sections()
            self._detect_corners()
            self._detect_obstacles()
    
    @computed_field
    @property
    def start(self) -> Coordinate:
        if not self.grids:
            return Coordinate(x=0, y=0)
        return self.grids[0].coords
    
    @computed_field
    @property
    def end(self) -> Coordinate:
        if not self.grids:
            return Coordinate(x=0, y=0)
        return self.grids[-1].coords
    
    @computed_field
    @property
    def length(self) -> float:
        return np.hypot(self.end.x - self.start.x, self.end.y - self.start.y)

    @property
    def angle(self) -> float:
        angle = np.arctan2(self.end.y - self.start.y, self.end.x - self.start.x)
        return np.degrees(angle)
    
    @property
    def has_a_corner(self) -> bool:
        return self.corners is not None
        
    def _calculate_sections(self) -> None:
        """
        Divide the path into sections
        Each section is created as its own Path object.
        """
        if not self.grids:
            return
            
        self.sections = []
        
        for i in range(0, len(self.grids)):
            
            
            section_grids = self.grids[i:i+5]
            section_cost = self.total_cost * (len(section_grids) / len(self.grids))
                
            section = Path(
                grids=section_grids,
                total_cost=section_cost,
                path_type="section",
            )
                
            self.sections.append(section)
    
    def _detect_corners(self) -> None:
        """
        Detect multiple corners by analyzing angle changes across multiple sections.
        Uses a variable window size to detect both sharp and gradual turns.
        """
        if not self.sections or len(self.sections) < 3:
            return
        
        self.corners = []
        
        def calculate_angle_between_sections(section1: Path, section2: Path) -> float:
            start1, end1 = section1.start, section1.end
            start2, end2 = section2.start, section2.end
            
            v1 = np.array(end1.to_tuple()) - np.array(start1.to_tuple())
            v2 = np.array(end2.to_tuple()) - np.array(start2.to_tuple())
            
            unit_v1 = v1 / np.linalg.norm(v1)
            unit_v2 = v2 / np.linalg.norm(v2)
            
            dot_product = np.dot(unit_v1, unit_v2)
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
            return np.degrees(angle)
        
        
        for i in range(len(self.sections) - 1):
            angle_change = calculate_angle_between_sections(self.sections[i], self.sections[i+1])
            
            if angle_change < 10 or angle_change > 80:
                continue
            
            sharpness = "sharp" if angle_change > 45 else "sweeping"
            cross_product = (
                (self.sections[i+1].end.x - self.sections[i].start.x) *
                (self.sections[i+1].end.y - self.sections[i].start.y) -
                (self.sections[i+1].end.y - self.sections[i].start.y) *
                (self.sections[i+1].end.x - self.sections[i].start.x)
            )
            
            corner_type = "right" if cross_product > 0 else "left"
            confidence = min(angle_change / 90, 1)
            
            corner = Corner(
                type=corner_type,
                sharpness=sharpness,
                confidence=confidence,
                angle_change=angle_change
            )
            self.corners.append(corner)
      
            
    def _detect_obstacles(self) -> None:
        pass