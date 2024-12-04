import cv2
import numpy as np

from grids import grids
from path_model import Path


grid_size = 20

def _draw_grid(frame, grid: dict, color: tuple[int, int, int]) -> None:
        """Draw a single grid on the frame."""
        x, y = grid.coords.x, grid.coords.y
        
        grid_corners = np.array([
            [x, y],
            [x + grid_size, y],
            [x + grid_size, y + grid_size],
            [x, y + grid_size]
        ], np.int32)
        
        cv2.fillPoly(frame, [grid_corners], color)
        
def _draw_section(frame, section: Path, color: tuple[int, int, int]) -> None:
    for grid in section.grids:
        _draw_grid(frame, grid, color)
    
    
for path in grids:
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)
    
    current_path = Path(
        grids=path,
        total_cost=100,
        path_type="path"
    )

    for grid in current_path.grids:
        _draw_grid(frame, grid, (255, 255, 255))
        
    section_colours = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255)
    ]
        
    for idx, section in enumerate(current_path.sections):
        _draw_section(frame, section, section_colours[idx % len(section_colours)])
        
    for corner in current_path.corners:
        x, y = corner.start.to_tuple()
        cv2.circle(frame, (x + (grid_size // 2), y + (grid_size // 2)), 5, (255, 255, 255), -1)
        cv2.putText(frame, f"{corner.direction}: {corner.sharpness} @ {corner.angle_change}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        

    cv2.imshow("Path", frame)
    cv2.waitKey(0)