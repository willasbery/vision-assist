import cv2
import numpy as np

from config import grid_size
from models import Coordinate, Grid
from utils import get_closest_grid_to_point, point_to_line_distance


# TODO:
# - Calculate the threshold for is_point_near_quadrilateral dynamically using frame dimensions

def find_corner_peak(binary: np.ndarray, region_mask: bool = None, return_min_max: bool = False) -> tuple:
    """
    Find the peak coordinate in a binary image, optionally within a specific region.
    
    Args:
        binary: Binary image
        region_mask: Mask for the region
        return_min_max: Return the minimum and maximum x-coordinates of the peak
        
    Returns:
        The peak coordinate in the binary image or None if no peak is found
    """
    if region_mask is not None:
        # Apply region mask to binary image
        binary = cv2.bitwise_and(binary, binary, mask=region_mask)
    
    # Get coordinates of white pixels
    white_pixels = np.where(binary == 255)
    if len(white_pixels[0]) == 0:
        return None
        
    y_coords = white_pixels[0]
    x_coords = white_pixels[1]
    
    # Find the minimum y-coordinate (highest point in the image)
    min_y = np.min(y_coords)
    
    # Get all x-coordinates that share this minimum y-coordinate
    peak_x_coords = x_coords[y_coords == min_y]
    
    # Find the middle x-coordinate
    if len(peak_x_coords) > 0:
        peak_x_coords = np.sort(peak_x_coords)
        middle_x = peak_x_coords[len(peak_x_coords) // 2]
        
        if return_min_max:
            return ((middle_x, min_y), (peak_x_coords[0], min_y), (peak_x_coords[-1], min_y))
        else:
            return (middle_x, min_y)
    
    return None


def get_protrusion_region(binary: np.ndarray, point: Coordinate, box_size: tuple[int, int] = (100, 100)) -> np.ndarray:
    """
    Create a mask for a square region around a point
    
    Args:
        binary: Binary image
        point: Centre of the region
        box_size: Size of the region
    
    Returns:
        A mask for the region
    
    """
    mask = np.zeros_like(binary)
    height, width = binary.shape
    
    # Calculate box coordinates
    box_height, box_width = box_size
    
    # Calculate box boundaries with image boundary checks
    x_start = max(0, point.x - box_width // 2)
    x_end = min(width, point.x + box_width // 2)
    y_start = max(0, point.y - box_height // 2)
    y_end = min(height, point.y + box_height // 2)
    
    # Create the mask
    mask[y_start:y_end, x_start:x_end] = 255
    
    return mask


def is_point_near_quadrilateral(point: Coordinate, quadrilateral: np.ndarray, threshold: float = 50.0) -> bool:
    """
    Check if a point is within a threshold distance of any edge.

    Args:
        point: The point
        quadrilateral: The quadrilateral
        threshold: The threshold distance
        
    Returns:
        True if the point is within the threshold distance of any edge, False otherwise
    """
    p1 = tuple(quadrilateral[0])
    p2 = tuple(quadrilateral[1])
    p3 = tuple(quadrilateral[2])
    p4 = tuple(quadrilateral[3])
    
    d1 = point_to_line_distance(point, p1, p2)
    d2 = point_to_line_distance(point, p2, p3)
    d3 = point_to_line_distance(point, p3, p1)
    d4 = point_to_line_distance(point, p1, p4)
    
    # Return True if any distance is less than threshold
    return min(d1, d2, d3, d4) < threshold


def check_full_column_below_point(point: Coordinate, grids: list[Grid]) -> bool:
    """
    Check if every cell in the column below a point is filled.
    
    Args:
        point: The point
        grids: The list of grids
        
    Returns:
        True if every cell in the column below the point is filled, False otherwise
    """
    closest_grid = get_closest_grid_to_point(point, grids)
    
    if not closest_grid:
        return False
    
    row = closest_grid.row
    col = closest_grid.col
    
    for i in range(row + 1, len(grids)):
        invalid_point = True
        
        for grid in grids[i]:
            if grid.col == col:
                invalid_point = False
                break
        
        if invalid_point:
            return False
    
    return True


def threshold_grids(frame: np.ndarray, grids: list[list[Grid]]) -> np.ndarray:
    """
    Create a black and white version of the frame to be used in protrusion detection.
    
    Args:
        frame: Frame to be thresholded
        grids: Grid information for the frame
    
    Returns:
        A black and white frame with the grids filled in white
    """
    height, width = frame.shape[:2]
    b_n_w_frame = np.zeros((height, width), dtype=np.uint8)
    
    for grid_row in grids:
        for grid in grid_row:
            i, j = grid.coords.x, grid.coords.y
            grid_corners = np.array([
                [i, j],
                [i + grid_size, j],
                [i + grid_size, j + grid_size],
                [i, j + grid_size]
            ], np.int32)
            
            cv2.fillPoly(b_n_w_frame, [grid_corners], 255)
            
    _, binary = cv2.threshold(b_n_w_frame, 127, 255, cv2.THRESH_BINARY)
    
    return binary


def find_protrusions(frame: np.ndarray, grids: list[list[Grid]]) -> list:
    """
    Find protrusions in the frame using grid information.
    Returns a list of protrusions if found, None otherwise.
    """
    thresholded_grids_frame = threshold_grids(frame, grids)
   
    # find global corner peak - highest point in the image that is on the path
    global_peak, global_peak_min_x, global_peak_max_x = find_corner_peak(thresholded_grids_frame, return_min_max=True)
   
    contours, _ = cv2.findContours(thresholded_grids_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    if len(contours) == 0:
        return None
   
    # find the largest contour, there should only ever be one contour as the segmentation returns the largest mask - but just in case
    contour = max(contours, key=cv2.contourArea)
    c_x, c_y, c_w, c_h = cv2.boundingRect(contour)
    
    hull = cv2.convexHull(contour, returnPoints=True)
    
    # OPTIMISATION: use the quadtrilateral formed by the hull to remove any paths falsely
    # detected as protrusions
    #   ---------------
    #  |               |
    #  |    C    D     |
    #  |    --*---     |
    #  |   /      \    |
    #  |  |        \   |
    #  |  /         \* | <-- Falsely detected protrusion
    #  | /           \ |
    # B|/             \|A
    #   ---------------
    # in the example above, we would remove the falsely detected protrusion
    # as it is within the threshold of the hull formed by the path (ABCD)
    hull_points = np.array([point[0] for point in hull])
    
    # SUB-OPTIMISATION: find the bottom left and bottom right points of the hull, by getting
    # the points with the smallest x and then the smallest y (for bottom left) but only consider
    # them if the column below them is filled, this way we get the actual bottom left
    #             ---------------
    #            |               |
    #            |               |
    #            |    ------     |
    #            |   /      \    |
    # Wrong ->   |  /        \   |   This one is wrong because it has the smallest x, 
    #            |  \         \  |   and then the smallest y, but the column below it is not filled
    #            |   \         \ |
    # Correct -> |   /          \|   Hence this is the correct bottom left
    #             ---------------
    
    # sort the hull points by the smallest x and smallest y coordinates
    bottom_left_candidates = hull_points[np.lexsort((hull_points[:, 1], hull_points[:, 0]))]
    # ensuring that the column below the point is filled
    bottom_left = None
    for point in bottom_left_candidates:
        if check_full_column_below_point(tuple(point), grids):
            bottom_left = tuple(point)
            break
    # if we cannot find a suitable bottom left point, we just take the first one
    if bottom_left is None:
        bottom_left = bottom_left_candidates[0]
        
    
    # sort the hull points by smallest y coordinate and largest x coordinate
    bottom_right_candidates = hull_points[np.lexsort((hull_points[:, 1], -hull_points[:, 0]))]
    
    bottom_right = None
    for point in bottom_right_candidates:
        if check_full_column_below_point(tuple(point), grids):
            bottom_right = tuple(point)
            break
        
    if bottom_right is None:
        bottom_right = bottom_right_candidates[0]
    
    # construct the quadrilateral encompassing the path as explained above
    quadrilateral_encompassing_path = np.array([
        bottom_left,
        bottom_right,
        global_peak_max_x,
        global_peak_min_x
    ])
   
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull_indices)
   
    protrusions = []
    processed_regions = []
    
    frame_height, frame_width = thresholded_grids_frame.shape[:2]
   
    for defect in defects:
        start, end, far, depth = defect[0]
        start = tuple(contour[start][0])
        end = tuple(contour[end][0])
        far = tuple(contour[far][0])
       
        v1 = np.array(start) - np.array(far)
        v2 = np.array(end) - np.array(far)
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle_deg = np.degrees(angle)
       
        is_significant = (
            depth > 0.35 * c_w and  
            angle_deg > 45 and      
            angle_deg < 120 and    
            start[1] < c_y + (0.7 * c_h)
        )
       
        if not is_significant:
            continue
       
        is_new_region = True
        min_distance = max(c_w // 4, 150)       
        for prev_start, prev_end in processed_regions:
            if (abs(start[0] - prev_start[0]) < min_distance and
                abs(start[1] - prev_start[1]) < min_distance):
                is_new_region = False
                break
           
        if is_new_region:
            box_height = frame_height // 8
            box_width = frame_width // 4
            region_mask = get_protrusion_region(thresholded_grids_frame, start, box_size=(box_height, box_width))
            peak = find_corner_peak(thresholded_grids_frame, region_mask)
           
            if peak is not None and peak != global_peak:
                if not is_point_near_quadrilateral(peak, quadrilateral_encompassing_path, threshold=50):
                    protrusions.append(peak)
                    processed_regions.append((start, end))
    
    return [global_peak] + protrusions if global_peak is not None else protrusions