import cv2
import numpy as np
from collections import defaultdict

from config import grid_size
from algorithms.misc.path_finding import a_star, calculate_row_penalty, get_penalty_colour
from corner_detection import calculate_path_direction, detect_corner
from path_detection import find_protrusions
from utils import get_closest_grid_to_point


def visualise_paths(frame: np.ndarray, paths: list[list], 
                   grids: list[list[dict]]) -> np.ndarray:
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
    
    # Create a set of path grid coordinates for faster lookup
    path_grid_coords = {
        (grid['coords']['x'], grid['coords']['y']) 
        for path in paths 
        for grid in path
    }
    
    # First draw non-path grids
    for grid_row in grids:
        for grid in grid_row:
            x, y = grid['coords']['x'], grid['coords']['y']
            
            if (x, y) in path_grid_coords:
                continue
                
            grid_corners = np.array([
                [x, y],
                [x + grid_size, y],
                [x + grid_size, y + grid_size],
                [x, y + grid_size]
            ], np.int32)
            
            colour = get_penalty_colour(grid['penalty'] or 0)
            cv2.fillPoly(frame, [grid_corners], colour)
    
    for path_idx, path in enumerate(paths):
        if not path:
            continue
            
        path_colors_dict = path_colors[path_idx % len(path_colors)]
        
        # Calculate path sections based on y coordinates
        y_coords = [grid['coords']['y'] for grid in path]
        min_y = min(y_coords)
        max_y = max(y_coords)
        y_range = abs(max_y - min_y)
        
        # Define thresholds for sections
        far_threshold = min_y + (y_range * 0.1)
        mid_threshold = min_y + (y_range * 0.4)
        
        # Categorize grids into sections
        far_grids = [grid for grid in path if grid['coords']['y'] <= far_threshold]
        mid_grids = [grid for grid in path if far_threshold < grid['coords']['y'] <= mid_threshold]
        close_grids = [grid for grid in path if grid['coords']['y'] > mid_threshold]
        
        # Draw path grids
        for grid in path:
            x, y = grid['coords']['x'], grid['coords']['y']
            grid_corners = np.array([
                [x, y],
                [x + grid_size, y],
                [x + grid_size, y + grid_size],
                [x, y + grid_size]
            ], np.int32)
            
            # Determine color based on grid section
            if grid in close_grids:
                colour = path_colors_dict['close']
            elif grid in mid_grids:
                colour = path_colors_dict['mid']
            else:
                colour = path_colors_dict['far']
                
            cv2.fillPoly(frame, [grid_corners], colour)
        
        # Draw path segments and add text if we have grids in all sections
        if far_grids and mid_grids and close_grids:
            # Extract center points for path direction calculation
            far_centers = [grid['centre'] for grid in far_grids]
            mid_centers = [grid['centre'] for grid in mid_grids]
            close_centers = [grid['centre'] for grid in close_grids]
            
            # Calculate path directions
            far_bottom, far_top, far_slope, far_length = calculate_path_direction(far_centers)
            mid_bottom, mid_top, mid_slope, mid_length = calculate_path_direction(mid_centers)
            close_bottom, close_top, close_slope, close_length = calculate_path_direction(close_centers)
            
            # Draw lines
            cv2.line(frame, far_bottom, far_top, [255, 255, 255], 2)
            cv2.line(frame, mid_bottom, mid_top, [255, 255, 255], 2)
            cv2.line(frame, close_bottom, close_top, [255, 255, 255], 2)
            
            # Add measurements text with offset
            text_offset = path_idx * 25
            
            # Helper function to avoid repetition in text rendering
            def add_path_text(text, position):
                cv2.putText(
                    frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, [255, 255, 255], 1, cv2.LINE_AA
                )
            
            # Add measurement texts
            add_path_text(
                f"Path {path_idx + 1} Far: {far_slope:.2f}, {far_length:.2f}",
                (far_bottom[0], far_bottom[1] + text_offset)
            )
            add_path_text(
                f"Path {path_idx + 1} Mid: {mid_slope:.2f}, {mid_length:.2f}",
                (mid_bottom[0], mid_bottom[1] + text_offset)
            )
            add_path_text(
                f"Path {path_idx + 1} Close: {close_slope:.2f}, {close_length:.2f}",
                (close_bottom[0], close_bottom[1] + text_offset)
            )
            
            # Detect and display corner information
            corner_type, confidence = detect_corner(close_slope, mid_slope, far_slope)
            if corner_type:
                cv2.putText(frame,
                    f"Path {path_idx + 1}: {corner_type.capitalize()} corner detected: {confidence:.2f}",
                    (30, 320 + path_idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 2, cv2.LINE_AA
                )
    
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
    
    grids: list[list[dict]] = [] # stores all grid dictionaries
    grid_lookup: dict[tuple[int, int], dict] = {}  # for fast coordinate lookups, used mainly in A*
    
    # Extract grid information from masks
    for result in results:
        if result.masks is None:
            continue
            
        for mask in result.masks.xy:
            points = np.int32([mask])
            x, y, w, h = cv2.boundingRect(points)
            
            # Ensure the grid is a multiple of 20
            if x % grid_size != 0:
                x = x - x % grid_size
            if y % grid_size != 0:
                y = y - y % grid_size
            if w % grid_size != 0:
                w = w + (grid_size - w % grid_size)
            if h % grid_size != 0:
                h = h + (grid_size - h % grid_size)
                
            row_count = 0
            for i in range(y, y + h, grid_size):
                col_count = 0
                this_row = []
                
                for j in range(x, x + w, grid_size):
                    grid_centre = (j + grid_size // 2, i + grid_size // 2)
                    
                    # Check if the grid centre is inside the mask
                    if cv2.pointPolygonTest(points, grid_centre, False) >= 0:
                        grid = {
                            "coords": {
                                "x": j,
                                "y": i
                            },
                            "centre": grid_centre,
                            "penalty": None,
                            "row": row_count,
                            "col": col_count
                        }
                        this_row.append(grid)
                        
                        # Add to lookup dictionary
                        grid_lookup[(j, i)] = grid
                        
                        col_count += 1
                
                if this_row:
                    grids.append(this_row)
                    row_count += 1
    
    # If there are no grids, return the original frame as we cannot extract any data
    if not grids:
        return frame
    
    # Calculate penalties for each grid
    for grid_row in grids:
        for grid in grid_row:
            penalty = calculate_row_penalty(grid, grids)
            grid["penalty"] = penalty
    
    # Create the graph to perform A* on later
    graph = defaultdict(list)
    for grid_row in grids:
        for grid in grid_row:
            x, y = grid["coords"]["x"], grid["coords"]["y"]
            current_pos = (x, y)
            
            # Define possible neighbor positions
            neighbor_positions = [
                (x + grid_size, y),  # right
                (x - grid_size, y),  # left
                (x, y + grid_size),  # down
                (x, y - grid_size)   # up
            ]
            
            # Use grid_lookup for efficient neighbor finding
            for nx, ny in neighbor_positions:
                if neighbor_grid := grid_lookup.get((nx, ny)):
                    dist = np.sqrt((x - nx)**2 + (y - ny)**2)
                    graph[current_pos].append(((nx, ny), dist))
    
    # Find the start grid to begin path finding from (the middle grid in the bottom row)
    bottom_row_grids = grids[-1]
    start_grid = bottom_row_grids[(len(bottom_row_grids) - 1) // 2]   
    
    # Find the protrusions, which will give us the end grids as these should be the ends of paths
    protrusion_peaks = find_protrusions(frame, grids)
    
    paths = []     
    for peak in protrusion_peaks:
        closest_grid = get_closest_grid_to_point(peak, grids)
        
        # using the grid_lookup for fast lookups speeds up the process by a lot
        path, _ = a_star(graph, start_grid, closest_grid, grid_lookup)
        # Convert coordinate tuples back to grid dictionaries for visualization
        grid_path = [grid_lookup[(path_node["coords"]["x"], path_node["coords"]["y"])] for path_node in path]
        paths.append(grid_path)
        
    # Visualize the paths on the frame
    frame = visualise_paths(frame, paths, grids)
    
    return frame