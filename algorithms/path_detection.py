import cv2
import numpy as np

from typing import List, Tuple

from config import grid_size


# TODO:
# - Add type hints to all functions
# - Further optimise the protrusion detection algorithm
# - Calculate the threshold for is_point_near_quadrilateral dynamically using frame dimensions



def find_corner_peak(binary: np.ndarray, region_mask: bool = None, return_min_max: bool = False) -> Tuple:
    """
    Find the peak coordinate in a binary image, optionally within a specific region
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


def get_protrusion_region(binary, point, box_size: Tuple[int, int] = (100, 100)) -> np.ndarray:
    """
    Create a mask for a square region around a point
    """
    mask = np.zeros_like(binary)
    height, width = binary.shape
    
    # Calculate box coordinates
    x, y = point
    box_height, box_width = box_size
    
    # Calculate box boundaries with image boundary checks
    x_start = max(0, x - box_width // 2)
    x_end = min(width, x + box_width // 2)
    y_start = max(0, y - box_height // 2)
    y_end = min(height, y + box_height // 2)
    
    # Create the mask
    mask[y_start:y_end, x_start:x_end] = 255
    
    return mask


def point_to_line_distance(point: Tuple[int, int], line_start: Tuple[int, int], line_end: Tuple[int, int]) -> float:
    """
    Calculate the perpendicular distance from a point to a line segment.
    """
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Calculate the distance
    numerator = abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1)
    denominator = np.sqrt((y2-y1)**2 + (x2-x1)**2)
    
    if denominator == 0:
        return np.sqrt((x-x1)**2 + (y-y1)**2)
    
    return numerator/denominator


def is_point_near_quadrilateral(point: Tuple[int, int], quadrilateral: np.ndarray, threshold: float = 50.0) -> bool:
    """
    Check if a point is within a threshold distance of any triangle edge.
    Returns True if point is near any edge, False otherwise.
    """
    # Get triangle vertices
    p1 = tuple(quadrilateral[0])
    p2 = tuple(quadrilateral[1])
    p3 = tuple(quadrilateral[2])
    p4 = tuple(quadrilateral[3])
    
    # Calculate distances to each edge
    d1 = point_to_line_distance(point, p1, p2)
    d2 = point_to_line_distance(point, p2, p3)
    d3 = point_to_line_distance(point, p3, p1)
    d4 = point_to_line_distance(point, p1, p4)
    
    # Return True if any distance is less than threshold
    return min(d1, d2, d3, d4) < threshold


def detect_protrusions(frame: np.ndarray, all_grids: List) -> List:
    """
    Detect protrusions in a binary or grayscale image.
    Uses stricter criteria to only detect significant protrusions.
    Filters out protrusions near quadrilateral edges.
    """
    frame_height, frame_width = frame.shape[:2]
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    _, binary = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
   
    # Find global corner peak - highest point in the image that is on the path
    global_peak, global_peak_min_x, global_peak_max_x = find_corner_peak(binary, return_min_max=True)
   
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    if len(contours) == 0:
        return None
   
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
    # In the example above, we would remove the falsely detected protrusion
    # as it is within the threshold of the hull formed by the path (ABCD)
    hull_points = np.array([point[0] for point in hull])
    
    # TODO: this can be further optimised, we ideally want the bottom left and bottom right points,
    # and these MUST have an entire column of cells below them until the frame_height, otherwise they are not valid
        
    # sort the hull points by the smallest x and smallest y coordinates
    bottom_left = hull_points[np.lexsort((hull_points[:, 1], hull_points[:, 0]))][0]
    
    # sort the hull points by smallest y coordinate and largest x coordinate
    bottom_right = hull_points[np.lexsort((hull_points[:, 1], -hull_points[:, 0]))][0]
    
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
    
    # for debug
    debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    cv2.polylines(debug_img, [quadrilateral_encompassing_path], True, (0, 255, 255), 2)
   
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
            region_mask = get_protrusion_region(binary, start, box_size=(box_height, box_width))
            peak = find_corner_peak(binary, region_mask)
           
            if peak is not None and peak != global_peak:
                if not is_point_near_quadrilateral(peak, quadrilateral_encompassing_path, threshold=50):
                    protrusions.append(peak)
                    processed_regions.append((start, end))
                    # Draw accepted peak in green
                    cv2.circle(debug_img, peak, 5, (0, 255, 0), -1)
                else:
                    # Draw rejected peak in red
                    cv2.circle(debug_img, peak, 5, (0, 0, 255), -1)
    
    # Draw global peak
    if global_peak:
        cv2.circle(debug_img, global_peak, 7, (255, 255, 0), -1)
    
    cv2.imshow("Protrusion Detection", debug_img)
    cv2.waitKey(1)
    
    return [global_peak] + protrusions


def threshold_grids(frame: np.ndarray, all_grids: List) -> np.ndarray:
    """Threshold the frame to show only the path."""
    # Create a single-channel (grayscale) frame
    height, width = frame.shape[:2]
    b_n_w_frame = np.zeros((height, width), dtype=np.uint8)
    
    for grid in all_grids:
        # Draw grid corners
        i, j = grid
        grid_corners = np.array([
            [i, j],
            [i + grid_size, j],
            [i + grid_size, j + grid_size],
            [i, j + grid_size]
        ], np.int32)
        
        # Fill the grid with white (255)
        cv2.fillPoly(b_n_w_frame, [grid_corners], 255)
    
    return b_n_w_frame


def find_protrusions(frame: np.ndarray, all_grids: List) -> List:
    """
    Find protrusions in the frame using grid information.
    Returns a list of protrusions if found, None otherwise.
    """
    # Create binary frame from grids
    binary_frame = threshold_grids(frame, all_grids)
   
    peaks = detect_protrusions(binary_frame, all_grids)
    return peaks