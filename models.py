from __future__ import annotations # for forward reference

import numpy as np
from pydantic import BaseModel, computed_field
from typing import Literal, Any

from config import grid_size


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
    direction: Literal["left", "right"]
    sharpness: Literal["sharp", "sweeping"]
    shape: Literal["inner", "outer", "optimal"]
    start: Coordinate
    end: Coordinate
    angle_change: float
    
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
    path_type: Literal["path", "section-straight", "section-curved"]
    sections: list[Path] | None = None  # a path section cannot have sections
    
    corners: list[Corner] | None = None  # a path section cannot have a corner 
    points: list[tuple[Coordinate, Coordinate]] | None = None
    
    test_data: dict = {}
    
    def model_post_init(self, __context: Any) -> None:
        if self.path_type == "path" and self.grids:  # only create sections for main path
            self._calculate_sections()
            self._detect_corners()
    
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
        Divide path into sections by identifying straight segments and the paths between them.
        A straight segment is defined as 3 or more grids moving only horizontally or vertically.
        """
        if not self.grids:
            return
            
        self.sections = []
        straight_sections: list[tuple[int, int]] = []  # [(start_idx, end_idx)]
        
        # First pass: identify straight sections
        current_start = 0
        last_direction = None
        straight_count = 1
        
        for i in range(1, len(self.grids)):
            current_grid = self.grids[i]
            prev_grid = self.grids[i-1]
            
            dx = current_grid.coords.x - prev_grid.coords.x
            dy = current_grid.coords.y - prev_grid.coords.y
            
            current_direction = "verticle" if dx == 0 and dy != 0 else \
                                "horizontal" if dy == 0 and dx != 0 else None
            last_direction = current_direction if i == 1 else last_direction
                
            if current_direction == last_direction:
                straight_count += 1
                if straight_count >= 5 and i == len(self.grids) - 1:
                    straight_sections.append((current_start, i))
            else:
                if straight_count >= 5:
                    straight_sections.append((current_start, i-1))
                current_start = i
                straight_count = 1
                
            last_direction = current_direction
        
        # Second pass: create sections from straight segments and paths between them
        last_end = 0
        for start, end in straight_sections:
            # add section for path between straight sections if it exists
            if start > last_end:
                between_grids = self.grids[last_end: start + 1] # include start grid for connectivity
                
                # if this section is less than 4 grids, add it to the previous section
                if len(between_grids) <= 4:
                    if self.sections:
                        prev_section = self.sections[-1]
                        # add the grids to the previous section
                        prev_section.grids.extend(between_grids[1:])
                        prev_section.total_cost = self.total_cost * (len(prev_section.grids) / len(self.grids))
                    else:
                        straight_grids = between_grids + self.grids[start: end + 1]
                        section_cost = self.total_cost * (len(straight_grids) / len(self.grids))
                        straight_section = Path(
                            grids=straight_grids,
                            total_cost=section_cost,
                            path_type="section-straight"
                        )
                        self.sections.append(straight_section)
                        last_end = end
                        continue
                else:
                    section_cost = self.total_cost * (len(between_grids) / len(self.grids))
                    between_section = Path(
                        grids=between_grids,
                        total_cost=section_cost,
                        path_type="section-curved"
                    )
                    self.sections.append(between_section)
            
            # before adding a new straight section, check if we can combine it with the previous section
            # we need to add this check because of the code that combines sections of less than 4 grids with
            # the previous section
            if self.sections and self.sections[-1].path_type == "section-straight":
                prev_section = self.sections[-1]
                # add new grids to the previous section (excluding the first grid to avoid duplication)
                straight_grids = self.grids[start: end + 1]
                prev_section.grids.extend(straight_grids[1:])
                prev_section.total_cost = self.total_cost * (len(prev_section.grids) / len(self.grids))
            else:
                straight_grids = self.grids[start: end + 1]
                section_cost = self.total_cost * (len(straight_grids) / len(self.grids))
                straight_section = Path(
                    grids=straight_grids,
                    total_cost=section_cost,
                    path_type="section-straight"
                )
                self.sections.append(straight_section)
                
            last_end = end
        
        # add final section if there are remaining grids
        if last_end < len(self.grids) - 1:
            final_grids = self.grids[last_end:]
            
            if len(final_grids) < 4 and self.sections:
                prev_section = self.sections[-1]
                prev_section.grids.extend(final_grids[1:])
                prev_section.total_cost = self.total_cost * (len(prev_section.grids) / len(self.grids))
            else:
                section_cost = self.total_cost * (len(final_grids) / len(self.grids))
                final_section = Path(
                    grids=final_grids,
                    total_cost=section_cost,
                    path_type="section-curved"
                )
                self.sections.append(final_section)
        
    def _get_closest_grid_to_point(self, point: Coordinate, grids: list[Grid]):
        """
        Get the closest grid to a point.
        
        Args:
            point: The point to find the closest grid to
            grids: The list of grids
        
        Returns:
            The closest grid to the point    
        """    
        closest_grid: dict = None
        min_distance = np.inf
        
        for grid in grids:
            if grid.empty:
                continue
            
            # Calculate euclidean distance
            distance = np.sqrt((point.x - grid.centre.x) ** 2 + (point.y - grid.centre.y) ** 2)
            
            if distance < min_distance:
                min_distance = distance
                closest_grid = grid

        return closest_grid
        
    def _detect_corners(self) -> None:
        """
        Detect corners on a non-straight section
        """
        if not self.sections:
            return
        
        self.corners = []
        self.points = []
        
        for section in self.sections:
            if section.start not in self.points:
                self.points.append(section.start)
            if section.end not in self.points:
                self.points.append(section.end)
                       
        
        for section in self.sections:
            # ignore straight sections
            if section.path_type == "section-straight":
                continue
            
            start_grid = section.grids[0]
            end_grid = section.grids[-1]
            
            dx = end_grid.centre.x - start_grid.centre.x
            dy = end_grid.centre.y - start_grid.centre.y
            angle_change = abs(np.degrees(np.arctan2(dy, dx)))
            
            direction = "right" if start_grid.centre.x - end_grid.centre.x < 0 else "left"
            
            midpoint = Coordinate(
                x=start_grid.centre.x + (dx // 2),
                y=start_grid.centre.y + (dy // 2) 
            )
        
            nearest_grid = self._get_closest_grid_to_point(midpoint, section.grids)
            euclidean_distance = np.hypot(abs(nearest_grid.centre.x - midpoint.x), abs(nearest_grid.centre.y - midpoint.y))
    
            # check if this nearest grid is to the left or right of the midpoint
            dy_midpoint_nearest = nearest_grid.centre.y - midpoint.y
            
            # this threshold calculation is very basic rn, it can be improved massively
            threshold = (np.hypot(dx, dy))**2 / (euclidean_distance + 1)**2
            
            if euclidean_distance < threshold:
                shape = "optimal"
            else:
                shape = "inner" if dy_midpoint_nearest < 0 else "outer"
                
            while angle_change > 90:
                angle_change -= 90
            
            sharpness = "sharp" if angle_change > 45 else "sweeping"
            
            corner = Corner(
                direction=direction,
                sharpness=sharpness,
                shape=shape,
                start=start_grid.coords,
                end=end_grid.coords,
                angle_change=angle_change,
                nearest_grid=nearest_grid,
                midpoint=midpoint,
                euclidean_distance=euclidean_distance
            )
            self.corners.append(corner)