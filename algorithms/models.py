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
    grids: list[Grid]
    total_cost: float
    section_type: Literal["close", "mid", "far"] | None = None # None for full path
    
    corner: Corner | None = None
    obstacles: list[Obstacle] | None = None
    
    far_section: Path | None = None
    mid_section: Path | None = None
    close_section: Path | None = None
    
    def model_post_init(self, __context: Any) -> None:
        if not self.section_type and self.grids:
            self._calculate_sections()
            self._detect_corner()
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
    def has_corner(self) -> bool:
        return self.corner is not None
    
    @computed_field
    @property
    def has_all_sections(self) -> bool:
        if self.section_type is not None:
            raise AttributeError("Section paths do not have sections")
        return (
            self.far_section is not None and 
            self.mid_section is not None and 
            self.close_section is not None
        )
        
    def _calculate_sections(self) -> None:
        """
        Calculate far, mid, and close sections of the path.
        Each section is created as its own Path object.
        """
        if not self.grids:
            return
            
        y_coords = [grid.coords.y for grid in self.grids]
        min_y = min(y_coords)
        max_y = max(y_coords)
        y_range = abs(max_y - min_y)
        
        far_threshold = min_y + (y_range * 0.1)  # <10% from the top
        mid_threshold = min_y + (y_range * 0.4)  # >10% and <40% from the top
        
        # Create section paths
        far_grids = [grid for grid in self.grids if grid.coords.y <= far_threshold]
        self.far_section = Path(
            grids=far_grids,
            total_cost=self.total_cost * (len(far_grids) / len(self.grids)),
            section_type='far'
        )
        
        mid_grids = [grid for grid in self.grids 
                    if far_threshold < grid.coords.y <= mid_threshold]
        self.mid_section = Path(
            grids=mid_grids,
            total_cost=self.total_cost * (len(mid_grids) / len(self.grids)),
            section_type='mid'
        )
        
        close_grids = [grid for grid in self.grids if grid.coords.y > mid_threshold]
        self.close_section = Path(
            grids=close_grids,
            total_cost=self.total_cost * (len(close_grids) / len(self.grids)),
            section_type='close'
        )
    
    def _normalize_slope(self, slope: float) -> float:
        return 1000 if abs(slope) == float('inf') else slope
    
    def _detect_corner(self) -> None:
        if not self.has_all_sections:
            return
        
        close_slope = self._normalize_slope(self.close_section.angle)
        mid_slope = self._normalize_slope(self.mid_section.angle)
        far_slope = self._normalize_slope(self.far_section.angle)
        
        diff_close_mid = abs(close_slope - mid_slope)
        diff_mid_far = abs(mid_slope - far_slope)
        total_angle_change = abs(far_slope - close_slope)
        max_slope = max(abs(close_slope), abs(mid_slope), abs(far_slope))
        
        confidence = min(diff_close_mid, diff_mid_far) / max_slope if max_slope > 0 else 0
        
        if abs(far_slope) > abs(mid_slope) > abs(close_slope):
            self.corner = Corner(
                type="left", 
                confidence=confidence, 
                angle_change=total_angle_change
            )
        elif abs(close_slope) > abs(mid_slope) > abs(far_slope):
            self.corner = Corner(
                type="right", 
                confidence=confidence, 
                angle_change=total_angle_change
            )
            
    def _detect_obstacles(self) -> None:
        pass    
    
# @dataclass
# class Path:
#     """
#     Represents a path through a grid with its properties and sections.
#     Each path can have far, mid, and close sections which are also Path objects.
#     """
#     grids: list[Grid]  # List of grid dictionaries in the path
#     total_cost: float  # Total cost/distance of the path
#     section_type: str | None = None  # One of: 'far', 'mid', 'close', or None for full path
#     obstacles: list[Obstacle] | None = None  # List of obstacles detected near or around the path
    
#     def __post_init__(self):
#         """Initialize section paths if this is a full path."""
#         self._far_section: Path | None = None
#         self._mid_section: Path | None = None
#         self._close_section: Path | None = None
#         self._corner: Corner | None = None
#         self._obastcles: list[Obstacle] | None = None
        
#         # Only calculate sections for full paths (non-section paths)
#         if not self.section_type and self.grids:
#             self._calculate_sections()
#             self._detect_corner()
#             self._detect_obstacles()
    
#     @property
#     def start(self) -> tuple[int, int]:
#         """Get the starting coordinates of the path."""
#         if not self.grids:
#             return (0, 0)
#         start_grid = self.grids[0]
#         return (start_grid.coords.x, start_grid.coords.y)
    
#     @property
#     def end(self) -> tuple[int, int]:
#         """Get the ending coordinates of the path."""
#         if not self.grids:
#             return (0, 0)
#         end_grid = self.grids[-1]
#         return (end_grid.coords.x, end_grid.coords.y)
    
#     @property
#     def grid_coords(self) -> list[tuple[int, int]]:
#         """Get list of grid coordinates in the path."""
#         return [(grid.coords.x, grid.coords.y) for grid in self.grids]
    
#     @property
#     def length(self) -> float:
#         """Calculate the length of the path."""
#         return np.hypot(self.end[0] - self.start[0], 
#                        self.end[1] - self.start[1])
    
#     @property
#     def angle(self) -> float:
#         """Calculate the angle of the path in degrees."""
#         delta_x = self.end[0] - self.start[0]
#         delta_y = self.end[1] - self.start[1]
#         angle = np.arctan2(delta_y, delta_x)
#         return np.degrees(angle)
    
#     @property
#     def has_corner(self) -> bool:
#         """Check if the path has a detected corner."""
#         return self._corner is not None
    
#     @property
#     def corner(self) -> Corner | None:
#         """Get the detected corner information if it exists."""
#         return self._corner
    
#     def _normalize_slope(self, slope: float) -> float:
#         """Normalize infinite slopes to a large finite value."""
#         return 1000 if abs(slope) == float('inf') else slope
    
#     def _detect_corner(self) -> None:
#         """
#         Detect if there's a corner in the path based on section slopes.
#         Updates the self._corner property with the detection result.
#         """
#         if not self.has_all_sections:
#             return
        
#         # Get and normalize slopes
#         close_slope = self._normalize_slope(self.close_section.angle)
#         mid_slope = self._normalize_slope(self.mid_section.angle)
#         far_slope = self._normalize_slope(self.far_section.angle)
        
#         # Calculate slope differences
#         diff_close_mid = abs(close_slope - mid_slope)
#         diff_mid_far = abs(mid_slope - far_slope)
#         total_angle_change = abs(far_slope - close_slope)
#         max_slope = max(abs(close_slope), abs(mid_slope), abs(far_slope))
        
#         # Calculate confidence based on both angle differences
#         confidence = min(diff_close_mid, diff_mid_far) / max_slope if max_slope > 0 else 0
        
#         # Detect corner type
#         if abs(far_slope) > abs(mid_slope) > abs(close_slope):
#             self._corner = Corner('left', confidence, total_angle_change)
#         elif abs(close_slope) > abs(mid_slope) > abs(far_slope):
#             self._corner = Corner('right', confidence, total_angle_change)
    
#     @property
#     def has_all_sections(self) -> bool:
#         """Check if the path has grids in all sections."""
#         if self.section_type:
#             raise AttributeError("Section paths don't have sub-sections")
#         return bool(self._far_section and self._far_section.grids and 
#                    self._mid_section and self._mid_section.grids and 
#                    self._close_section and self._close_section.grids)
    
#     def _calculate_sections(self) -> None:
#         """
#         Calculate far, mid, and close sections of the path.
#         Each section is created as its own Path object.
#         """
#         if not self.grids:
#             return
            
#         y_coords = [grid.coords.y for grid in self.grids]
#         min_y = min(y_coords)
#         max_y = max(y_coords)
#         y_range = abs(max_y - min_y)
        
#         far_threshold = min_y + (y_range * 0.1)  # <10% from the top
#         mid_threshold = min_y + (y_range * 0.4)  # >10% and <40% from the top
        
#         # Create section paths
#         far_grids = [grid for grid in self.grids if grid.coords.y <= far_threshold]
#         self._far_section = Path(
#             grids=far_grids,
#             total_cost=self.total_cost * (len(far_grids) / len(self.grids)),
#             section_type='far'
#         )
        
#         mid_grids = [grid for grid in self.grids 
#                     if far_threshold < grid.coords.y <= mid_threshold]
#         self._mid_section = Path(
#             grids=mid_grids,
#             total_cost=self.total_cost * (len(mid_grids) / len(self.grids)),
#             section_type='mid'
#         )
        
#         close_grids = [grid for grid in self.grids if grid.coords.y > mid_threshold]
#         self._close_section = Path(
#             grids=close_grids,
#             total_cost=self.total_cost * (len(close_grids) / len(self.grids)),
#             section_type='close'
#         )
    
#     @property
#     def far_section(self) -> 'Path':
#         """Get the far section of the path."""
#         if self.section_type:
#             raise AttributeError("Section paths don't have sub-sections")
#         return self._far_section or Path([], float('inf'), 'far')
    
#     @property
#     def mid_section(self) -> 'Path':
#         """Get the mid section of the path."""
#         if self.section_type:
#             raise AttributeError("Section paths don't have sub-sections")
#         return self._mid_section or Path([], float('inf'), 'mid')
    
#     @property
#     def close_section(self) -> 'Path':
#         """Get the close section of the path."""
#         if self.section_type:
#             raise AttributeError("Section paths don't have sub-sections")
#         return self._close_section or Path([], float('inf'), 'close')
    
#     def _detect_obstacles(self) -> None:
#         pass