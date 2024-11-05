import cv2
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict

from config import (
    grid_size, 
    penalty_colour_gradient,
    close_grid_colour, 
    mid_grid_colour, 
    far_grid_colour
)
from algorithms.misc.path_finding_old import a_star, calculate_row_penalty
from corner_detection import calculate_path_direction, detect_corner
from path_detection import find_protrusions


def get_penalty_colour(penalty: float) -> Tuple[int, int, int]:
    """Get the colour based on the penalty value."""
    closest_key = min(penalty_colour_gradient.keys(), key=lambda x: abs(x - penalty))
    return penalty_colour_gradient[closest_key]


def visualise_paths(frame: np.ndarray, paths: List[List], 
                   all_grids: List, penalties_dict: Dict) -> np.ndarray:
    """
    Visualize multiple detected paths on the frame.
    
    Args:
        frame: Input frame
        paths: List of paths, where each path is a list of grid coordinates
        all_grids: List of all grid coordinates
        penalties_dict: Dictionary of penalties for each grid
    
    Returns:
        Frame with visualizations
    """
    if not paths:
        return frame
        
    # Define distinct colors for different paths
    path_colors = [
        {'close': (0, 255, 0), 'mid': (0, 200, 0), 'far': (0, 150, 0)},  # Green variants
        {'close': (255, 0, 0), 'mid': (200, 0, 0), 'far': (150, 0, 0)},  # Red variants
        {'close': (0, 0, 255), 'mid': (0, 0, 200), 'far': (0, 0, 150)},  # Blue variants
        {'close': (255, 255, 0), 'mid': (200, 200, 0), 'far': (150, 150, 0)}  # Yellow variants
    ]
    
    # First draw non-path grids
    path_grids = set()
    for path in paths:
        path_grids.update(path)
        
    for grid in all_grids:
        if grid not in path_grids:
            i, j = grid
            grid_corners = np.array([
                [i, j],
                [i + grid_size, j],
                [i + grid_size, j + grid_size],
                [i, j + grid_size]
            ], np.int32)
            colour = get_penalty_colour(penalties_dict[grid])
            cv2.fillPoly(frame, [grid_corners], colour)
    
    # Then draw each path with its own color scheme
    for path_idx, path in enumerate(paths):
        if not path:
            continue
            
        path_colors_dict = path_colors[path_idx % len(path_colors)]
        
        # Calculate path sections
        min_y = min(path, key=lambda g: g[1])[1]
        max_y = max(path, key=lambda g: g[1])[1]
        y_range = abs(max_y - min_y)
        
        far_grids = [grid for grid in path if grid[1] <= min_y + (y_range * 0.1)]
        mid_grids = [grid for grid in path if min_y + (y_range * 0.1) < grid[1] <= min_y + (y_range * 0.4)]
        close_grids = [grid for grid in path if grid[1] > min_y + (y_range * 0.4)]
        
        # Draw path grids
        for grid in path:
            i, j = grid
            grid_corners = np.array([
                [i, j],
                [i + grid_size, j],
                [i + grid_size, j + grid_size],
                [i, j + grid_size]
            ], np.int32)
            
            if grid in close_grids:
                colour = path_colors_dict['close']
            elif grid in mid_grids:
                colour = path_colors_dict['mid']
            else:
                colour = path_colors_dict['far']
                
            cv2.fillPoly(frame, [grid_corners], colour)
        
        # Draw path segments and add text
        if far_grids and mid_grids and close_grids:
            far_bottom, far_top, far_slope, far_length = calculate_path_direction(far_grids)
            mid_bottom, mid_top, mid_slope, mid_length = calculate_path_direction(mid_grids)
            close_bottom, close_top, close_slope, close_length = calculate_path_direction(close_grids)
            
            # Draw lines with white color for visibility
            cv2.line(frame, far_bottom, far_top, [255, 255, 255], 2)
            cv2.line(frame, mid_bottom, mid_top, [255, 255, 255], 2)
            cv2.line(frame, close_bottom, close_top, [255, 255, 255], 2)
            
            # Add text with offset based on path index to avoid overlap
            text_offset = path_idx * 25
            cv2.putText(frame, f"Path {path_idx + 1} Far: {far_slope:.2f}, {far_length:.2f}", 
                       (far_bottom[0], far_bottom[1] + text_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1, cv2.LINE_AA)
            cv2.putText(frame, f"Path {path_idx + 1} Mid: {mid_slope:.2f}, {mid_length:.2f}", 
                       (mid_bottom[0], mid_bottom[1] + text_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1, cv2.LINE_AA)
            cv2.putText(frame, f"Path {path_idx + 1} Close: {close_slope:.2f}, {close_length:.2f}", 
                       (close_bottom[0], close_bottom[1] + text_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1, cv2.LINE_AA)
            
            # Detect corner type
            corner_type, confidence = detect_corner(close_slope, mid_slope, far_slope)
            if corner_type:
                cv2.putText(frame, f"Path {path_idx + 1}: {corner_type.capitalize()} corner detected: {confidence:.2f}", 
                           (30, 320 + path_idx * 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 2, cv2.LINE_AA)
    
    return frame


def process_frame(frame: np.ndarray, model) -> np.ndarray:
    """
    Process a single frame with path detection and visualization.
    
    Args:
        frame: Input frame
        model: YOLO model instance
    
    Returns:
        Processed frame with visualizations
    """
    results = model.predict(frame, conf=0.5)
        
    all_grids = [] # stores all grid coordinates in a 1D list
    grid_centres = {}
    penalties_dict = {}
    row_dict = defaultdict(list) # stores the grids in each row
    frame_height, frame_width = frame.shape[:2]
    
    # Extract grid information from masks
    for result in results:
        if result.masks is None:
            continue
            
        for mask in result.masks.xy:
            points = np.int32([mask])
            x, y, w, h = cv2.boundingRect(points)
            
            min_x = max(x, 0)
            min_y = max(y, 0)
            
            max_x = min(x + w, frame_width - grid_size)
            max_y = min(y + h, frame_height - grid_size)
            
            for i in range(min_y, max_y, grid_size):

                for j in range(min_x, max_x, grid_size):
                    grid_centre = (j + grid_size // 2, i + grid_size // 2)
                    
                    # Check if the grid centre is inside the mask
                    if cv2.pointPolygonTest(points, grid_centre, False) >= 0:
                        grid = (j, i)
                        all_grids.append(grid)
                        grid_centres[grid] = grid_centre
                        row_dict[grid_centre[1]].append(grid)
                       
    
    # If there are no grids, return the original frame as we cannot extract any data
    if not all_grids:
        return frame
        
    # Calculate penalties
    for _, grids_in_row in row_dict.items():
        min_x = min(grids_in_row, key=lambda g: g[0])[0]
        max_x = max(grids_in_row, key=lambda g: g[0])[0]
        
        for grid in grids_in_row:
            x, _ = grid
            penalties_dict[grid] = calculate_row_penalty(x, min_x, max_x)    
    
    # Create graph
    graph = defaultdict(list)
    for grid in all_grids:
        x, y = grid
        neighbours = [
            (x + grid_size, y),
            (x - grid_size, y),
            (x, y + grid_size),
            (x, y - grid_size)
        ]
        
        for neighbour in neighbours:
            if neighbour in grid_centres:
                dist = np.linalg.norm(np.array(grid_centres[grid]) - np.array(grid_centres[neighbour]))
                graph[grid].append((neighbour, dist))
                
    # Find start and end points
    bottom_row_grids = [grid for grid in all_grids if grid[1] == max(all_grids, key=lambda g: g[1])[1]]
    bottom_row_x_values = [grid[0] for grid in bottom_row_grids]
    bottom_row_centre_x = sum(bottom_row_x_values) // len(bottom_row_x_values)
    
    start_grid = min(bottom_row_grids, key=lambda grid: abs(grid[0] - bottom_row_centre_x))
    
    # Find protrusions, giving us our end points
    protrusion_peaks = find_protrusions(frame, all_grids)
    
    if protrusion_peaks:
        paths = []
        for peak in protrusion_peaks:
            # Find the closest grid to the peak
            closest_grid = min(all_grids, key=lambda grid: np.linalg.norm(np.array(grid_centres[grid]) - np.array(peak)))
        
            path, _ = a_star(graph, start_grid, closest_grid, penalties_dict)
            paths.append(path)
        
        path_visualisation = visualise_paths(frame, paths, all_grids, penalties_dict)
    else:
        path_visualisation = visualise_paths(frame, [], all_grids, penalties_dict)
    
    return path_visualisation