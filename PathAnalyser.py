import cv2
import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Literal, Optional

from config import grid_size
from models import PathColours, Path


class PathAnalyser:
    _instance: ClassVar[Optional['PathAnalyser']] = None
    _initialized: bool = False
    
    PATH_COLORS = [
        PathColours(close=(0, 255, 0), mid=(0, 200, 0), far=(0, 150, 0)),  # Green variants
        PathColours(close=(255, 0, 0), mid=(200, 0, 0), far=(150, 0, 0)),  # Red variants
        PathColours(close=(0, 0, 255), mid=(0, 0, 200), far=(0, 0, 150)),  # Blue variants
        PathColours(close=(255, 255, 0), mid=(200, 200, 0), far=(150, 150, 0))  # Yellow variants
    ]
    
    def __new__(cls):
        """Ensure only one instance of PathAnalyser exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the path detector only once."""
        if not self._initialized:
            self._initialized = True
            self.frame: np.ndarray| None = None
            self.paths: list[Path] = []
            
    def _draw_path_grid(self, grid: dict, color: tuple[int, int, int]) -> None:
        """Draw a single grid on the frame."""
        x, y = grid.coords.x, grid.coords.y
        
        grid_corners = np.array([
            [x, y],
            [x + grid_size, y],
            [x + grid_size, y + grid_size],
            [x, y + grid_size]
        ], np.int32)
        
        cv2.fillPoly(self.frame, [grid_corners], color)
    
    def _draw_path_sections(self, path: Path, path_colors: PathColours, path_idx: int) -> None:
        """Draw path sections with measurements and corner detection."""
        for grid in path.far_section.grids:
            self._draw_path_grid(grid, path_colors.far)
        for grid in path.mid_section.grids:
            self._draw_path_grid(grid, path_colors.mid)
        for grid in path.close_section.grids:
            self._draw_path_grid(grid, path_colors.close)
        
        if path.has_all_sections:
            # Draw connecting lines
            for section in [path.far_section, path.mid_section, path.close_section]:
                cv2.line(self.frame, section.start.to_tuple(), section.end.to_tuple(), (255, 255, 255), 2)
            
            # Add measurements text
            text_offset = path_idx * 25
            for section, name in [
                (path.far_section, "Far"),
                (path.mid_section, "Mid"),
                (path.close_section, "Close")
            ]:
                cv2.putText(
                    self.frame,
                    f"Path {path_idx + 1} {name}: {section.angle:.2f}, {section.length:.2f}",
                    (section.start.x, section.start.y + text_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                )
            
            if path.has_corner:
                cv2.putText(
                    self.frame,
                    f"Path {path_idx + 1}: {path.corner.type.capitalize()} corner detected: {path.corner.confidence:.2f}",
                    (30, 320 + path_idx * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                )
            
    def __call__(self, frame: np.ndarray, paths: list[Path]) -> np.ndarray:
        """Process the frame and visualize all paths."""
        self.frame = frame
        self.paths = paths
        
        for idx, path in enumerate(self.paths):
            path_colors = self.PATH_COLORS[idx % len(self.PATH_COLORS)]
            self._draw_path_sections(path, path_colors, idx)
                
        return self.frame


# Create singleton for export
path_analyser = PathAnalyser()