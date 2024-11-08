import cv2
import numpy as np
from typing import ClassVar, Optional

from config import grid_size
from models import Grid, Peak, ConvexityDefect, Coordinate
from utils import get_closest_grid_to_point, point_to_line_distance


class ProtrusionDetector:
    """
    ProtrusionDetector is a class that detects protrusions in a given image, in the hopes of finding paths.
    """
    _instance: ClassVar[Optional['ProtrusionDetector']] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the detector only once."""
        if not self._initialized:
            self._initialized = True
            self.frame: np.ndarray | None = None
            self.grids: list[list[Grid]] | None = None
            self.height: int = 0
            self.width: int = 0
            self.binary: np.ndarray | None = None
        
    def _create_binary_image(self) -> np.ndarray:
        binary = np.zeros((self.height, self.width), dtype=np.uint8)
        
        for grid_row in self.grids:
            for grid in grid_row:
                if grid.empty:
                    continue
                
                x, y = grid.coords.to_tuple()
                
                corners = np.array([
                    [x, y],
                    [x + grid_size, y],
                    [x + grid_size, y + grid_size],
                    [x, y + grid_size]
                ], np.int32)
                
                cv2.fillPoly(binary, [corners], 255)
        
        return cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)[1]
    
    def _find_peak(
        self, 
        defect_centre: Coordinate | None = None, 
        # grid_lookup: dict[tuple[int, int], Grid], 
        region_around_protrusion: np.ndarray | None = None # TODO: remove the optional region mask
    ) -> list[Peak] | None:
        """    
        TODO:
        Try to find the "peak" of a path, in either the vertical, left or right direction.
        The peak is the highest point in the path, and is used to determine the path's direction.
        
        Here are the three cases (the * indicates the peak):
        
               Vertical             Left               Right
             -------------      -------------      -------------
            |             |    |        ---- |    | ----        |
            |   --*---    |    |     ---     |    |     ---     |
            |  |      |   |    |   *|        |    |        |*   |
            | /        \  |    |     ---     |    |    ----     |
            |/          \ |    |        ---  |    | ---         |
             -------------      -------------      -------------     
             
        """        
        if region_around_protrusion is None or defect_centre is None:
            region = self.binary
            x_offset, y_offset = 0, 0
        else:
            region = region_around_protrusion
            # Calculate offsets based on the region position in the full frame
            box_height, box_width = region.shape[:2]
            x_offset = max(0, defect_centre.x - box_width // 2)
            y_offset = max(0, defect_centre.y - box_height // 2)
            
        region_points = np.where(region == 255)
        if not region_points[0].size:
            return []
        
        y_coords, x_coords = region_points
        min_y = np.min(y_coords)
        peak_x_coords = np.sort(x_coords[y_coords == min_y])
        
        if not peak_x_coords.size:
            return []
        
        gaps = np.diff(peak_x_coords)
        # needs to be at least 1/4 of the grid size apart to account for noise
        split_points = np.where(gaps > (grid_size // 4))[0] + 1
        contiguous_groups = np.split(peak_x_coords, split_points)
        
        # Find the peak(s) in the vertical direction if the region mask is None
        if region_around_protrusion is None:
            peaks = []
            
            for group in contiguous_groups:
                centre_x = int(group[len(group) // 2])
                peaks.append(Peak(
                    centre=Coordinate(x=centre_x, y=int(min_y)),
                    left=Coordinate(x=int(group[0]), y=int(min_y)),
                    right=Coordinate(x=int(group[-1]), y=int(min_y)),
                    orientation="up" # no region mask defaults to vertical
                ))
                
            return peaks
            
        # Calculate the point distribution relative to centre
        centre_x = region.shape[1] // 2
        left_points = np.sum(x_coords < centre_x)
        right_points = np.sum(x_coords > centre_x)
        total_points = left_points + right_points
        
        left_ratio = left_points / total_points if total_points > 0 else 0
        right_ratio = right_points / total_points if total_points > 0 else 0
        
        centre = None
        left = None
        right = None
        
        if abs(left_ratio - right_ratio) < 0.2:
            orientation = "up"
            # Add offsets to convert to global coordinates
            centre = Coordinate(x=centre_x + x_offset, y=int(min_y) + y_offset)
            left = Coordinate(x=int(peak_x_coords[0]) + x_offset, y=int(min_y) + y_offset)
            right = Coordinate(x=int(peak_x_coords[-1]) + x_offset, y=int(min_y) + y_offset)
            
        elif left_ratio > right_ratio:
            orientation = "left"
            leftmost_x = np.min(x_coords)
            leftmost_y_coords = y_coords[x_coords == leftmost_x]
            
            middle_y = int(leftmost_y_coords[len(leftmost_y_coords) // 2])
            max_y = np.max(leftmost_y_coords)
            min_y = np.min(leftmost_y_coords)
            
            # Add offsets to convert to global coordinates
            centre = Coordinate(x=int(leftmost_x) + x_offset, y=middle_y + y_offset)
            left = Coordinate(x=int(leftmost_x) + x_offset, y=int(max_y) + y_offset)
            right = Coordinate(x=int(leftmost_x) + x_offset, y=int(min_y) + y_offset)
            
        else:
            orientation = "right"
            rightmost_x = np.max(x_coords)
            rightmost_y_coords = y_coords[x_coords == rightmost_x]
            
            middle_y = int(rightmost_y_coords[len(rightmost_y_coords) // 2])
            max_y = np.max(rightmost_y_coords)
            min_y = np.min(rightmost_y_coords)
            
            # Add offsets to convert to global coordinates
            centre = Coordinate(x=int(rightmost_x) + x_offset, y=middle_y + y_offset)
            left = Coordinate(x=int(rightmost_x) + x_offset, y=int(min_y) + y_offset)
            right = Coordinate(x=int(rightmost_x) + x_offset, y=int(max_y) + y_offset)

        # Validate that all points were found
        if centre is None or left is None or right is None:
            return []

        # Visualize the peak (using local coordinates for visualization)
        debug_img = region.copy()
        # Convert back to local coordinates for visualization
        local_centre = Coordinate(x=centre.x - x_offset, y=centre.y - y_offset)
        local_left = Coordinate(x=left.x - x_offset, y=left.y - y_offset)
        local_right = Coordinate(x=right.x - x_offset, y=right.y - y_offset)
        
        cv2.circle(debug_img, local_centre.to_tuple(), 5, 255, -1)
        cv2.circle(debug_img, local_left.to_tuple(), 3, 128, -1)
        cv2.circle(debug_img, local_right.to_tuple(), 3, 128, -1)
        cv2.putText(debug_img, orientation, (local_centre.x - 20, local_centre.y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)

        cv2.imshow("Peak", debug_img)
        cv2.waitKey(0)
                
        return [Peak(
            centre=centre,
            left=left,
            right=right,
            orientation=orientation
        )]
        
    def _get_region_around_protrusion(self, point: Coordinate) -> np.ndarray:
        box_height, box_width = self.frame.shape[1] // 6, self.frame.shape[0] // 6
    
        # Calculate crop boundaries in original image
        x_start = max(0, point.x - box_width // 2)
        x_end = min(self.width, point.x + box_width // 2)
        y_start = max(0, point.y - box_height // 2)
        y_end = min(self.height, point.y + box_height // 2)
        
        # Create empty box
        box = np.zeros((box_height, box_width), dtype=np.uint8)
        
        # Get the cropped region from binary image
        binary_cropped = self.binary[y_start:y_end, x_start:x_end]
        
        # Calculate where in the box the cropped region should go
        # If x_start or y_start is 0, it means we're at the image edge
        # and should position the crop at the box edge
        box_x_start = 0 if x_start == 0 else (box_width // 2) - (point.x - x_start)
        box_y_start = 0 if y_start == 0 else (box_height // 2) - (point.y - y_start)
        
        # Calculate end positions
        box_x_end = box_x_start + binary_cropped.shape[1]
        box_y_end = box_y_start + binary_cropped.shape[0]
        
        # Ensure we don't exceed box dimensions
        if box_x_end > box_width:
            binary_cropped = binary_cropped[:, :-(box_x_end - box_width)]
            box_x_end = box_width
        if box_y_end > box_height:
            binary_cropped = binary_cropped[:-(box_y_end - box_height), :]
            box_y_end = box_height
            
        # Place the cropped region in the correct position in the box
        box[box_y_start:box_y_end, box_x_start:box_x_end] = binary_cropped
        
        return box
    
    def _is_valid_bottom_point(self, point: Coordinate) -> bool:
        """Check if a point has a complete column of grids below it."""
        closest_grid = get_closest_grid_to_point(point, self.grids)
        if not closest_grid:
            return False
            
        for row in self.grids[closest_grid.row + 1:]:
            if row[closest_grid.col].empty:
                return False
        return True
    
    def _is_point_near_quadrilateral(self, point: Coordinate, quad: np.ndarray, threshold: int) -> bool:
        distances = [
            point_to_line_distance(
                point, 
                quad[i],
                quad[(i + 1) % 4]
            )
            for i in range(len(quad) - 1)
        ]
        
        return min(distances) < threshold
    
    def _get_quadrilateral(self, global_peaks: list[Peak], contour: np.ndarray) -> list[Coordinate]:
        """Get the quadrilateral that encompasses the main path."""
        hull_points = cv2.convexHull(contour, returnPoints=True)
        hull_points = np.array([point[0] for point in hull_points])
        
        # Find bottom left point
        left_candidates = hull_points[np.lexsort((hull_points[:, 1], hull_points[:, 0]))]
        left_candidates = [Coordinate(x=int(point[0]), y=int(point[1])) for point in left_candidates]
        
        bottom_left = next(
            (point for point in left_candidates if self._is_valid_bottom_point(point)),
            left_candidates[0]
        )
        
        # Find bottom right point
        right_candidates = hull_points[np.lexsort((hull_points[:, 1], -hull_points[:, 0]))]
        right_candidates = [Coordinate(x=int(point[0]), y=int(point[1])) for point in right_candidates]
        bottom_right = next(
            (point for point in right_candidates if self._is_valid_bottom_point(point)),
            right_candidates[0]
        )
        
        # TODO: CHECK IF THIS OPTIMISATION IS NECESSARY
        # If the left and right are super close, widen the quadrilateral to at least half as wide as the frame
        if abs(bottom_right.x - bottom_left.x) < self.width // 2:
            how_much_to_widen = (self.width // 2) - abs(bottom_right.x - bottom_left.x)
            # find the position across the frame width of each point
            left_ratio = bottom_left.x / (self.width // 2)
            right_ratio = (bottom_right.x - (self.width // 2)) / (self.width // 2)
            
            # if the quadrilateral is skewed to the right, move the right point less than the right
            if right_ratio > left_ratio:
                bottom_right.x = min(self.width, bottom_right.x + how_much_to_widen * 0.4)
                bottom_left.x = max(0, bottom_left.x - how_much_to_widen * 0.6)
            else:
                bottom_right.x = min(self.width, bottom_right.x + how_much_to_widen * 0.6)
                bottom_left.x = max(0, bottom_left.x - how_much_to_widen * 0.4)      
        # END OF OPTIMISATION      
        
        return [
            bottom_left,
            bottom_right,
            max(global_peaks, key=lambda peak: peak.right.x).right,
            min(global_peaks, key=lambda peak: peak.left.x).left
        ]
        
    def _filter_protrusions(
        self, 
        protrusions: list[Coordinate], 
        convex_hull: cv2.Mat, 
        distance_threshold: int = 100
    ) -> list[Coordinate]:
        """ 
        Filter out protrusions that are too close to each other.
        """
        if not protrusions:
            return []
        
        def euclidean_distance(p1: Coordinate, p2: Coordinate) -> float:
            return np.linalg.norm(np.array(p1.to_tuple()) - np.array(p2.to_tuple()))
        
        clusters: list[list[Coordinate]] = []
        
        for point in protrusions:
            added_to_cluster = False
            
            for cluster in clusters:
                if any(
                    euclidean_distance(point, cluster_point) < distance_threshold 
                    for cluster_point in cluster
                ):
                    cluster.append(point)
                    added_to_cluster = True
                    break    
            
            if not added_to_cluster:
                clusters.append([point])
                
        # Get the best point from each cluster, by comparing its distance to the convex hull
        filtered_protrusions = []
        
        for cluster in clusters:
            best_point = min(cluster, key=lambda point: cv2.pointPolygonTest(convex_hull, point.to_tuple(), True))
            filtered_protrusions.append(best_point)  
            
        return filtered_protrusions      
        
        
    def __call__(self, frame: np.ndarray, grids: list[list[Grid]], grid_lookup: dict[tuple[int, int], Grid]) -> list[Coordinate]:
        """
        Process a new frame to detect protrusions.
        
        Returns:
            tuple: (list of protrusion coordinates, debug visualization image)
        """
        self.frame = frame
        self.grids = grids
        self.height, self.width = frame.shape[:2]
        self.binary = self._create_binary_image()
        
        # Create debug image (copy of original frame or blank canvas)
        debug_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Draw binary image in gray
        debug_image[self.binary == 255] = [255, 255, 255]
        
        # Find global peak
        global_peaks = self._find_peak()
        if global_peaks is None:
            print("No global peak found.")
            return []
            
        # Process contours
        contours, _ = cv2.findContours(self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("No contours found.")
            return []
            
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw convex hull in green
        hull = cv2.convexHull(contour)
        cv2.drawContours(debug_image, [hull], -1, (0, 255, 0), 2)
        
        quad = self._get_quadrilateral(global_peaks, contour)
        
        # Draw quadrilateral in blue
        quad_points = np.array([[p.x, p.y] for p in quad], dtype=np.int32)
        cv2.polylines(debug_image, [quad_points], True, (255, 0, 0), 2)
        
        # Process defects
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull_indices)
        
        print("Len defects:", len(defects))
        
        if defects is None:
            return [global_peak.centre for global_peak in global_peaks]
        
        protrusions = []
      
        for defect in defects:
            convexity_defect = ConvexityDefect(
                start=Coordinate(x=int(contour[defect[0][0]][0][0]), y=int(contour[defect[0][0]][0][1])),
                end=Coordinate(x=int(contour[defect[0][1]][0][0]), y=int(contour[defect[0][1]][0][1])),
                far=Coordinate(x=int(contour[defect[0][2]][0][0]), y=int(contour[defect[0][2]][0][1])),
                depth=float(defect[0][3])
            )
            
            if not (convexity_defect.depth > 0.35 * w and 
                45 < convexity_defect.angle_degrees < 120 and
                convexity_defect.start.y < y + (0.7 * h)):
                continue     
            
            # Get the centre of the region around the defect
            defect_centre = Coordinate(
                x=(convexity_defect.start.x + convexity_defect.end.x) // 2,
                y=(convexity_defect.start.y + convexity_defect.end.y) // 2
            )
                
            region_around_protrusion = self._get_region_around_protrusion(defect_centre)
            peaks = self._find_peak(defect_centre, region_around_protrusion)
            
            for peak in peaks:
                if peak and not self._is_point_near_quadrilateral(peak.centre, quad, threshold=50) and cv2.pointPolygonTest(quad_points, peak.centre.to_tuple(), False) < 0:
                    protrusions.append(peak.centre)
                    # Draw protrusion peak in red
                    cv2.circle(debug_image, (peak.centre.x, peak.centre.y), 5, (0, 0, 255), -1)
                
        filtered_protrusions = self._filter_protrusions(protrusions, hull)
        
        # DEBUG
        # Add legend
        legend_y = 30
        # Draw global peak last so it's on top
        for global_peak in global_peaks:
            cv2.circle(debug_image, (global_peak.centre.x, global_peak.centre.y), 8, (255, 0, 255), -1)  # magenta
            
        cv2.putText(debug_image, "Binary Image: Gray", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        cv2.putText(debug_image, "Convex Hull: Green", (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(debug_image, "Quadrilateral: Blue", (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(debug_image, "Defect Points: Yellow", (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(debug_image, "Far Points: Cyan", (10, legend_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(debug_image, "Protrusions: Red", (10, legend_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(debug_image, "Global Peak: Magenta", (10, legend_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        cv2.imshow("Protrusion Detection", debug_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # END OF DEBUG
        
        print("Length of global peaks:", len(global_peaks))
        return [global_peak.centre for global_peak in global_peaks] + filtered_protrusions