import numpy as np

from config import grid_size



def calculate_path_direction(path: list[tuple[int, int]]) -> tuple:
    """
    Calculate the general direction of the path.
    """
    bottom_grid = max(path, key=lambda g: g[1])
    top_grid = min(path, key=lambda g: g[1])
    
    bottom_x = bottom_grid[0]
    top_x = top_grid[0]
    
    centre_bottom = (bottom_x + (grid_size // 2), bottom_grid[1] + (grid_size // 2))
    centre_top = (top_x + (grid_size // 2), top_grid[1] + (grid_size // 2))
    
    delta_x = centre_top[0] - centre_bottom[0]
    delta_y = centre_top[1] - centre_bottom[1]
    slope = delta_y / delta_x if delta_x != 0 else float('inf')
    length = np.hypot(delta_x, delta_y)
    
    return centre_bottom, centre_top, slope, length


def detect_corner(close_slope: float, mid_slope: float, far_slope: float) -> tuple[str, float]:
    """
    Detect if there's a corner based on the slopes
    """
    close_slope = 1000 if abs(close_slope) == float('inf') else close_slope
    mid_slope = 1000 if abs(mid_slope) == float('inf') else mid_slope
    far_slope = 1000 if abs(far_slope) == float('inf') else far_slope
    
    diff_close_mid = abs(close_slope - mid_slope)
    diff_mid_far = abs(mid_slope - far_slope)
    
    if abs(far_slope) > abs(mid_slope) > abs(close_slope):
        confidence = min(diff_close_mid, diff_mid_far) / max(abs(close_slope), abs(mid_slope), abs(far_slope))
        return 'left', confidence
    
    elif abs(close_slope) > abs(mid_slope) > abs(far_slope):
        confidence = min(diff_close_mid, diff_mid_far) / max(abs(close_slope), abs(mid_slope), abs(far_slope))
        return 'right', confidence
    
    return None, 0