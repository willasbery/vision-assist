import numpy as np
import cv2
from typing import List, Tuple, Dict
from config import grid_size

def find_corner_peak(binary, region_mask=None):
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
        return (middle_x, min_y)
    
    return None

def get_protrusion_region(binary, point, box_size=200):
    """
    Create a mask for a square region around a point
    """
    mask = np.zeros_like(binary)
    height, width = binary.shape
    
    # Calculate box coordinates
    x, y = point
    half_size = box_size // 2
    
    # Calculate box boundaries with image boundary checks
    x_start = max(0, x - half_size)
    x_end = min(width, x + half_size)
    y_start = max(0, y - half_size)
    y_end = min(height, y + half_size)
    
    # Create the mask
    mask[y_start:y_end, x_start:x_end] = 255
    
    return mask

# def detect_protrusions(frame: np.ndarray) -> Dict:
#     """
#     Detect protrusions in a binary or grayscale image.
#     """
#     # Ensure we're working with a grayscale image
#     if len(frame.shape) == 3:
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Threshold if not already binary
#     _, binary = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
    
#     # Find global corner peak
#     global_peak = find_corner_peak(binary)
    
#     # Method 1: Harris Corner Detection
#     corners = cv2.cornerHarris(binary.astype(np.float32), blockSize=2, ksize=3, k=0.04)
#     corners = cv2.dilate(corners, None)
    
#     # Method 2: Contour Analysis
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if len(contours) > 0:
#         contour = contours[0]
        
#         # Find convex hull
#         hull = cv2.convexHull(contour)
        
#         # Find convexity defects
#         hull_indices = cv2.convexHull(contour, returnPoints=False)
#         defects = cv2.convexityDefects(contour, hull_indices)
        
#         protrusion_data = []
#         if defects is not None:
#             for i in range(defects.shape[0]):
#                 s, e, f, d = defects[i, 0]
#                 start = tuple(contour[s][0])
#                 end = tuple(contour[e][0])
#                 far = tuple(contour[f][0])
                
#                 # Filter significant protrusions based on depth
#                 if d > 1000:  # Adjust threshold as needed
#                     # Get region mask using 200x200 box around the start point
#                     region_mask = get_protrusion_region(binary, start, box_size=200)
                    
#                     # Find peak for this protrusion
#                     peak = find_corner_peak(binary, region_mask)
                    
#                     protrusion_data.append({
#                         'start': start,
#                         'end': end,
#                         'far': far,
#                         'depth': d,
#                         'peak': peak,
#                         'region_mask': region_mask
#                     })
        
#         return {
#             'corners': corners,
#             'contour': contour,
#             'hull': hull,
#             'protrusions': protrusion_data,
#             'global_peak': global_peak
#         }
    
#     return None


def detect_protrusions(frame: np.ndarray) -> Dict:
    """
    Detect protrusions in a binary or grayscale image.
    Only detects distinct protrusions based on angle and depth criteria.
    Always includes global peak in results.
    """
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Threshold if not already binary
    _, binary = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
    
    # Find global corner peak
    global_peak = find_corner_peak(binary)
    
    # Method 1: Harris Corner Detection
    corners = cv2.cornerHarris(binary.astype(np.float32), blockSize=2, ksize=3, k=0.04)
    corners = cv2.dilate(corners, None)
    
    # Method 2: Contour Analysis
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    contour = contours[0]
    hull = cv2.convexHull(contour)
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull_indices)
    
    protrusion_data = []
    
    # Always add global peak as a protrusion if it exists
    if global_peak is not None:
        protrusion_data.append({
            'start': global_peak,
            'end': global_peak,
            'far': global_peak,
            'depth': 0,  # Using 0 since this is the highest point
            'peak': global_peak,
            'region_mask': None,
            'angle': 0  # Using 0 since this is the highest point
        })
    
    if defects is not None:
        min_y = min(contour[:, 0, 1])
        max_y = max(contour[:, 0, 1])
        height = max_y - min_y
        processed_regions = []
        
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            v1 = np.array(start) - np.array(far)
            v2 = np.array(end) - np.array(far)
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angle_deg = np.degrees(angle)
            
            min_depth = height * 0.1
            
            is_significant = (
                d > min_depth and
                angle_deg < 120 and
                start[1] < max_y - height * 0.4
            )
            
            if is_significant:
                is_new_region = True
                for prev_start, prev_end in processed_regions:
                    if (abs(start[0] - prev_start[0]) < 100 and 
                        abs(start[1] - prev_start[1]) < 100):
                        is_new_region = False
                        break
                
                if is_new_region:
                    region_mask = get_protrusion_region(binary, start, box_size=200)
                    peak = find_corner_peak(binary, region_mask)
                    
                    if peak is not None and peak != global_peak:
                        protrusion_data.append({
                            'start': start,
                            'end': end,
                            'far': far,
                            'depth': d,
                            'peak': peak,
                            'region_mask': region_mask,
                            'angle': angle_deg
                        })
                        processed_regions.append((start, end))
    
    return {
        'corners': corners,
        'contour': contour,
        'hull': hull,
        'protrusions': protrusion_data,
        'global_peak': global_peak
    }


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
    
    # Detect protrusions in the binary frame
    results = detect_protrusions(binary_frame)
    
    if results is not None:
        # Visualize results
        vis_img = visualize_results(binary_frame, results)
        
        # Print peaks information
        print(f"Global peak coordinate: {results['global_peak']}")
        for i, protrusion in enumerate(results['protrusions']):
            print(f"Protrusion {i+1} peak: {protrusion['peak']}")
        
        # Save and display results
        cv2.imwrite("./protrusion_detection.png", vis_img)
        cv2.imshow("Protrusion Detection", vis_img)
        cv2.waitKey(1)  # Changed from 0 to 1 to not block video processing
        
        return [result['peak'] for result in results['protrusions']]
    
    return None

    # protrusion_peaks = [result['peak'] for result in results['protrusions']]
    # return protrusion_peaks
        


def visualize_results(img, results):
    """
    Visualize detection results on the image.
    """
    # Ensure input is grayscale
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    
    # Create BGR image for visualization
    vis_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    
    # # Draw corners
    # vis_img[results['corners'] > 0.01 * results['corners'].max()] = [0, 0, 255]
    
    # Draw contour
    cv2.drawContours(vis_img, [results['contour']], -1, (0, 255, 0), 2)
    
    # Draw convex hull
    cv2.drawContours(vis_img, [results['hull']], -1, (255, 0, 0), 2)
    
    # Draw protrusions and their peaks
    colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 128)]
    for i, protrusion in enumerate(results['protrusions']):
        color = colors[i % len(colors)]
        
        # Draw protrusion point
        cv2.circle(vis_img, protrusion['start'], 5, color, -1)
        
        # Draw box region
        x, y = protrusion['start']
        half_size = 100  # 200/2
        cv2.rectangle(vis_img, 
                     (max(0, x - half_size), max(0, y - half_size)),
                     (min(vis_img.shape[1], x + half_size), min(vis_img.shape[0], y + half_size)),
                     color, 1)
        
        # Draw protrusion peak if exists
        if protrusion['peak'] is not None:
            x, y = protrusion['peak']
            cv2.circle(vis_img, (x, y), 5, color, -1)
            cv2.line(vis_img, (x-10, y), (x+10, y), color, 2)
            cv2.line(vis_img, (x, y-10), (x, y+10), color, 2)
    
    # Draw global peak coordinate
    if results['global_peak'] is not None:
        x, y = results['global_peak']
        cv2.circle(vis_img, (x, y), 7, (255, 255, 255), -1)
        cv2.line(vis_img, (x-15, y), (x+15, y), (255, 255, 255), 2)
        cv2.line(vis_img, (x, y-15), (x, y+15), (255, 255, 255), 2)
    
    return vis_img