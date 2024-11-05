import cv2
import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Optional

from config import grid_size
from utils import get_closest_grid_to_point, point_to_line_distance


@dataclass
class ConvexityDefect:
    """ Represents a convexity defect in a contour. """
    start: tuple[int, int]
    end: tuple[int, int]
    far: tuple[int, int]
    depth: float
    
    @property
    def angle_degrees(self) -> float:
        """Calculate the angle of the defect in degrees."""
        v1 = np.array(self.start) - np.array(self.far)
        v2 = np.array(self.end) - np.array(self.far)
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return np.degrees(angle)


@dataclass
class Peak:
    """ Represents a peak point with its coordinates and boundaries. """
    center: tuple[int, int]
    left: tuple[int, int] | None = None
    right: tuple[int, int] | None = None


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
            self.frame: Optional[np.ndarray] = None
            self.grids: Optional[list[list[dict]]] = None
            self.height: int = 0
            self.width: int = 0
            self.binary: Optional[np.ndarray] = None
        
    def _create_binary_image(self) -> np.ndarray:
        binary = np.zeros((self.height, self.width), dtype=np.uint8)
        
        for grid_row in self.grids:
            for grid in grid_row:
                x, y = grid["coords"]["x"], grid["coords"]["y"]
                corners = np.array([
                    [x, y],
                    [x + grid_size, y],
                    [x + grid_size, y + grid_size],
                    [x, y + grid_size]
                ], np.int32)
                cv2.fillPoly(binary, [corners], 255)
        
        return cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)[1]
    
    def _find_peak(self, region_mask: np.ndarray | None = None) -> tuple[int, int] | None:
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
            center=(peak_x_coords[len(peak_x_coords) // 2], min_y),
            left=(peak_x_coords[0], min_y),
            right=(peak_x_coords[-1], min_y)
        )
        
    def _get_protrusion_region(self, point: tuple[int, int]) -> np.ndarray:
        box_height = self.height // 8
        box_width = self.width // 4
        mask = np.zeros_like(self.binary)
        
        x_start = max(0, point[0] - box_width // 2)
        x_end = min(self.width, point[0] + box_width // 2)
        y_start = max(0, point[1] - box_height // 2)
        y_end = min(self.height, point[1] + box_height // 2)
        
        mask[y_start:y_end, x_start:x_end] = 255
        return mask
    
    def _is_valid_bottom_point(self, point: tuple[int, int]) -> bool:
        """Check if a point has a complete column of grids below it."""
        closest_grid = get_closest_grid_to_point(point, self.grids)
        if not closest_grid:
            return False
            
        col = closest_grid["col"]
        for row in self.grids[closest_grid["row"] + 1:]:
            if not any(grid["col"] == col for grid in row):
                return False
        return True
    
    def _is_point_near_quadrilateral(self, point: tuple[int, int], quad: np.ndarray, threshold: int) -> bool:
        distances = [
            point_to_line_distance(point, tuple(quad[i]), tuple(quad[(i + 1) % 4]))
            for i in range(4)
        ]
        
        return min(distances) < threshold

    
    def _get_quadrilateral(self, global_peak: Peak, contour: np.ndarray) -> np.ndarray:
        """Get the quadrilateral that encompasses the main path."""
        hull_points = cv2.convexHull(contour, returnPoints=True)
        hull_points = np.array([point[0] for point in hull_points])
        
        # Find bottom left point
        left_candidates = hull_points[np.lexsort((hull_points[:, 1], hull_points[:, 0]))]
        bottom_left = next(
            (tuple(point) for point in left_candidates if self._is_valid_bottom_point(tuple(point))),
            tuple(left_candidates[0])
        )
        
        # Find bottom right point
        right_candidates = hull_points[np.lexsort((hull_points[:, 1], -hull_points[:, 0]))]
        bottom_right = next(
            (tuple(point) for point in right_candidates if self._is_valid_bottom_point(tuple(point))),
            tuple(right_candidates[0])
        )
        
        return np.array([
            bottom_left,
            bottom_right,
            global_peak.right,
            global_peak.left
        ])
        
    def __call__(self, frame: np.ndarray, grids: list[list[dict]]) -> list[tuple[int, int]]:
        """
        Process a new frame to detect protrusions.
        """
        self.frame = frame
        self.grids = grids
        self.height, self.width = frame.shape[:2]
        self.binary = self._create_binary_image()
        
        # Find global peak
        global_peak = self._find_peak()
        if global_peak is None:
            return []
            
        # Process contours
        contours, _ = cv2.findContours(self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
            
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        quad = self._get_quadrilateral(global_peak, contour)
        
        # Process defects
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull_indices)
        if defects is None:
            return [global_peak.center]
        
        # Find protrusions
        processed_regions = set()
        protrusions = []
        min_distance = max(w // 4, 150)
        
        for defect in defects:
            convexity_defect = ConvexityDefect(
                start=tuple(contour[defect[0][0]][0]),
                end=tuple(contour[defect[0][1]][0]),
                far=tuple(contour[defect[0][2]][0]),
                depth=defect[0][3]
            )
            
            if not (convexity_defect.depth > 0.35 * w and 
                   45 < convexity_defect.angle_degrees < 120 and
                   convexity_defect.start[1] < y + (0.7 * h)):
                continue
                
            defect_key = (convexity_defect.start, convexity_defect.end)
            if defect_key in processed_regions:
                continue
                
            is_new_region = all(
                abs(convexity_defect.start[0] - prev[0][0]) >= min_distance or
                abs(convexity_defect.start[1] - prev[0][1]) >= min_distance
                for prev in processed_regions
            )
            
            if not is_new_region:
                continue
                
            region_mask = self._get_protrusion_region(convexity_defect.start)
            peak = self._find_peak(region_mask)
            
            if peak and peak.center != global_peak.center:
                if not self._is_point_near_quadrilateral(peak.center, quad, threshold=50):
                    protrusions.append(peak.center)
                    processed_regions.add(defect_key)
        
        return [global_peak.center] + protrusions