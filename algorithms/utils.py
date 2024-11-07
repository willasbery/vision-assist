import numpy as np


def get_closest_grid_to_point(point: tuple[int, int], grids: list[list[dict]]):
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
    
    for grid_row in grids:
        for grid in grid_row:
            if grid.empty:
                continue
            
            centre_x, centre_y = grid.centre
            
            # Calculate euclidean distance
            distance = np.sqrt((point[0] - centre_x) ** 2 + (point[1] - centre_y) ** 2)
            
            if distance < min_distance:
                min_distance = distance
                closest_grid = grid
    
    return closest_grid


def point_to_line_distance(point: tuple[int, int], line_start: tuple[int, int], line_end: tuple[int, int]) -> float:
    """
    Calculate the perpendicular distance from a point to a line segment.
    
    Args:
        point: The point
        line_start: The start of the line segment
        line_end: The end of the line segment
    
    Returns:
        The perpendicular distance from the point to the line segment
    """
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Calculate the distance
    numerator = abs((y2 -y1 ) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    
    if denominator == 0:
        return np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    
    return numerator / denominator