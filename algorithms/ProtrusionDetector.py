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
    
    def _find_peak(self, grid_lookup: dict[tuple[int, int], Grid], region_mask: np.ndarray | None = None) -> Peak | None:
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
             
        Args:
            grid_lookup: The lookup table for grids
            region_mask: The region mask to apply to the binary image
        
        Returns:
            The peak of the path
        """
        
        working_binary = cv2.bitwise_and(self.binary, self.binary, mask=region_mask) \
            if region_mask is not None else self.binary
            
        white_pixels = np.where(working_binary == 255)
        if not white_pixels[0].size:
            return None
        
        y_coords, x_coords = white_pixels
        
        min_y = np.min(y_coords)
        
        peak_x_coords = np.sort(x_coords[y_coords == min_y])
        
        if not peak_x_coords.size:
            return None
            
        return Peak(
            center=Coordinate(x=int(peak_x_coords[len(peak_x_coords) // 2]), y=int(min_y)),
            left=Coordinate(x=int(peak_x_coords[0]), y=int(min_y)),
            right=Coordinate(x=int(peak_x_coords[-1]), y=int(min_y))
        )
        
    def _get_protrusion_region(self, point: Coordinate) -> np.ndarray:
        box_height = self.height // 8
        box_width = self.width // 4
        mask = np.zeros_like(self.binary)
        
        x_start = max(0, point.x - box_width // 2)
        x_end = min(self.width, point.x + box_width // 2)
        y_start = max(0, point.y - box_height // 2)
        y_end = min(self.height, point.y + box_height // 2)
        
        mask[y_start:y_end, x_start:x_end] = 255
        return mask
    
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
    
    def _get_quadrilateral(self, global_peak: Peak, contour: np.ndarray) -> list[Coordinate]:
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
        
        return [
            bottom_left,
            bottom_right,
            global_peak.right,
            global_peak.left
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
        global_peak = self._find_peak(grid_lookup)
        if global_peak is None:
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
        
        quad = self._get_quadrilateral(global_peak, contour)
        
        # Draw quadrilateral in blue
        quad_points = np.array([[p.x, p.y] for p in quad], dtype=np.int32)
        cv2.polylines(debug_image, [quad_points], True, (255, 0, 0), 2)
        
        # Process defects
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull_indices)
        
        if defects is None:
            return [global_peak.center]
        
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
                
            region_mask = self._get_protrusion_region(convexity_defect.start)
            peak = self._find_peak(region_mask)
            
            if peak and peak.center != global_peak.center:
                if not self._is_point_near_quadrilateral(peak.center, quad, threshold=50):
                    protrusions.append(peak.center)
                    # Draw protrusion peak in red
                    cv2.circle(debug_image, (peak.center.x, peak.center.y), 5, (0, 0, 255), -1)
                
        filtered_protrusions = self._filter_protrusions(protrusions, hull)
        
        # DEBUG
        # Add legend
        legend_y = 30
        # Draw global peak last so it's on top
        cv2.circle(debug_image, (global_peak.center.x, global_peak.center.y), 8, (255, 0, 255), -1)  # magenta
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
        
        return [global_peak.center] + filtered_protrusions