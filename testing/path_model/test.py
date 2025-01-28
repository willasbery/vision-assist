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
    frame[:] = (255, 255, 255)
    
    current_path = Path(
        grids=path,
        total_cost=100,
        path_type="path"
    )

    # for grid in current_path.grids:
    #     _draw_grid(frame, grid, (255, 255, 255))
        
    section_colours = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255)
    ]
        
    for idx, section in enumerate(current_path.sections):
        _draw_section(frame, section, section_colours[idx % len(section_colours)])
        
    cv2.imwrite(f"{path[-1].coords.x}_{path[-1].coords.y}_sections.png", frame)
        
    for corner in current_path.corners:
        x, y = corner.start.to_tuple()
        cv2.circle(frame, (x + (grid_size // 2), y + (grid_size // 2)), 5, (0, 0, 0), -1)
        cv2.putText(frame, f"{corner.direction} & {corner.shape}: {corner.sharpness} @ {corner.angle_change}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.circle(frame, (corner.midpoint.x, corner.midpoint.y), 5, (0, 0, 0), -1)
        
        _draw_grid(frame, corner.nearest_grid, (0, 0, 0))
        cv2.putText(frame, f"{corner.euclidean_distance}", corner.midpoint.to_tuple(), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
    
    for point in current_path.points:
        x, y = point.to_tuple()
        cv2.circle(frame, (x + (grid_size // 2), y + (grid_size // 2)), 5, (0, 0, 0), -1)
        
    for i in range(0, len(current_path.points) - 1):
        start, end = current_path.points[i], current_path.points[i+1]
        x1, y1 = start.to_tuple()
        x2, y2 = end.to_tuple()
        
        x1 += grid_size // 2
        y1 += grid_size // 2
        x2 += grid_size // 2
        y2 += grid_size // 2
        
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)


    cv2.imwrite(f"{path[-1].coords.x}_{path[-1].coords.y}_nearest_grid.png", frame)
        
    
    frame = cv2.resize(frame, (576, 1024), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Path", frame)
    cv2.waitKey(0)
    