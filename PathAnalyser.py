import cv2
import numpy as np
from typing import ClassVar, Optional

from config import grid_size
from models import Corner, PathColours, Path


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
            self.frame: np.ndarray | None = None
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
    
    def _draw_corner_marker(self, section_idx: int, corner: Corner, path: Path) -> None:
        """Draw a corner marker and label at the corner location."""
        # Retrieve the section where the corner is detected
        # section = path.sections[section_idx]

        # # Calculate the midpoint of the section to position the marker
        # corner_x = (section.start.x + section.end.x) // 2
        # corner_y = (section.start.y + section.end.y) // 2

        # # Define marker properties
        # marker_size = 10
        # marker_color = (255, 255, 255)  # Yellow

        # # Draw the corner marker
        # cv2.drawMarker(
        #     self.frame,
        #     (corner_x, corner_y),
        #     marker_color,
        #     cv2.MARKER_DIAMOND,
        #     marker_size,
        #     2
        # )

        # # Prepare corner information text
        # corner_text = (
        #     f"{corner.type.capitalize()} {corner.sharpness.capitalize()} "
        #     f"{corner.angle_change:.1f} Conf: {corner.confidence:.2f}"
        # )

        # # Draw the corner information text
        # cv2.putText(
        #     self.frame,
        #     corner_text,
        #     (corner_x, corner_y - 10),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     marker_color,
        #     2,
        #     cv2.LINE_AA
        # )
        pass
   
    def _draw_path_sections(self, path: Path, path_colors: PathColours, path_idx: int) -> None:
        """Draw path sections with measurements and corner detection."""
        if not path.sections:
            return
       
        # Draw grids for each section
        for i, section in enumerate(path.sections):
            # Calculate color based on section position
            section_progress = i / len(path.sections)
            if section_progress < 0.33:
                color = path_colors.far
            elif section_progress < 0.66:
                color = path_colors.mid
            else:
                color = path_colors.close
                
            # Draw section grids
            for grid in section.grids:
                self._draw_path_grid(grid, color)
           
        # Draw connecting lines between sections
        for section in path.sections:
            start_x = section.start.x + (grid_size // 2)
            start_y = section.start.y + (grid_size // 2)
            end_x = section.end.x + (grid_size // 2)
            end_y = section.end.y + (grid_size // 2)
            cv2.line(self.frame, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)
        
        # Draw corners if detected
        if path.corners:
            for idx, corner in enumerate(path.corners):
                self._draw_corner_marker(idx, corner, path)
           
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