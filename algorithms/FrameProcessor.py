import cv2
import numpy as np
from collections import defaultdict
from pydantic import BaseModel
from typing import ClassVar, Optional, List, Dict, Tuple
from ultralytics import YOLO

from config import grid_size
from models import Grid, Coordinate
from PathFinding import path_finder, penalty_calculator
from PathAnalyser import Path, path_analyser
from ProtrusionDetector import ProtrusionDetector
from utils import get_closest_grid_to_point


class FrameProcessor:
    """
    FrameProcessor handles the processing of video frames for path detection.
    """
    _instance: ClassVar[Optional['FrameProcessor']] = None
    _initialized: bool = False
    
    def __new__(cls, model: YOLO, verbose: bool) -> 'FrameProcessor':  
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model: YOLO, verbose: bool) -> None:
        """Initialize the processor only once."""
        if not self._initialized:
            self._initialized = True
            self.model = model
            self.verbose = verbose
            
            self.frame: Optional[np.ndarray] = None
            self.grids: list[list[Grid]] = []
            self.grid_lookup: dict[tuple[int, int], Grid] = {}
            self.protrusion_detector = ProtrusionDetector()
            
    def _reject_blurry_frames(self, frame: np.ndarray) -> bool:
        """Reject blurry frames based on the Laplacian variance."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_level = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
        return blur_level < 100
    
    def _extract_grid_information(self, results) -> None:
        """Extract grid information from YOLO detection results."""
        self.grids.clear()
        self.grid_lookup.clear()
        
        for result in results:
            if result.masks is None:
                continue
                
            for mask in result.masks.xy:
                points = np.int32([mask])
                x, y, w, h = cv2.boundingRect(points)
                
                # Ensure the grid is a multiple of grid_size
                x = x - (x % grid_size)
                y = y - (y % grid_size)
                w = w + (grid_size - w % grid_size) if w % grid_size != 0 else w
                h = h + (grid_size - h % grid_size) if h % grid_size != 0 else h
                
                row_count = 0
                for i in range(y, y + h, grid_size):
                    col_count = 0
                    this_row = []
                    
                    for j in range(x, x + w, grid_size):
                        grid_centre = (j + grid_size // 2, i + grid_size // 2)
                        
                        grid = Grid(
                            coords=Coordinate(x=j, y=i), 
                            centre=grid_centre, 
                            penalty=None, 
                            row=row_count, 
                            col=col_count, 
                            empty=False if cv2.pointPolygonTest(points, grid_centre, False) >= 0 else True
                        )
                        
                        this_row.append(grid)
                        self.grid_lookup[(j, i)] = grid
                        col_count += 1
                    
                    self.grids.append(this_row)
                    row_count += 1
    
    def _calculate_penalties(self) -> None:
        """Calculate penalties for each grid."""
        for grid_row in self.grids:
            for grid in grid_row:
                if grid.empty:
                    continue
                
                grid.penalty = penalty_calculator.calculate_row_penalty(grid, self.grids)
    
    def _create_graph(self) -> defaultdict:
        """Create a graph for A* pathfinding."""
        graph = defaultdict(list)
        for grid_row in self.grids:
            for grid in grid_row:
                if grid.empty:
                    continue
                
                x, y = grid.coords.x, grid.coords.y
                current_pos = (x, y)
                
                neighbor_positions = [
                    (x + grid_size, y),  # right
                    (x - grid_size, y),  # left
                    (x, y + grid_size),  # down
                    (x, y - grid_size)   # up
                ]
                
                for nx, ny in neighbor_positions:
                    if neighbor_grid := self.grid_lookup.get((nx, ny)):
                        dist = np.sqrt((x - nx)**2 + (y - ny)**2)
                        graph[current_pos].append(((nx, ny), dist))
        
        return graph
    
    def _find_paths(self, protrusion_peaks: List[Tuple[int, int]], graph: defaultdict) -> List[Path]:
        """Find paths using PathFinder."""
        paths = []
        if not self.grids:
            return paths
            
        # Find the start grid (middle grid in the bottom row)
        bottom_row_grids = self.grids[-1]
        start_grid = bottom_row_grids[(len(bottom_row_grids) - 1) // 2]
        
        for peak in protrusion_peaks:
            closest_grid = get_closest_grid_to_point(peak, self.grids)
            grid_path, total_cost = path_finder.find_path(graph, start_grid, closest_grid, self.grid_lookup)
            
            if grid_path:  # Only create Path object if a valid path was found
                path = Path(grids=grid_path, total_cost=total_cost)
                paths.append(path)
        
        return paths
    
    def _draw_non_path_grids(self) -> None:
        """Draw all non-path grids with their penalty colors."""
        # Get all path grid coordinates
        path_grid_coords = set()
        for grid_row in self.grids:
            for grid in grid_row:
                if grid.empty:
                    continue
                
                coords = (grid.coords.x, grid.coords.y)
                if coords not in path_grid_coords:
                    self._draw_grid(grid, penalty_calculator.get_penalty_colour(grid.penalty or 0))
    
    def _draw_grid(self, grid: Dict, color: Tuple[int, int, int]) -> None:
        """Draw a single grid on the frame."""
        x, y = grid.coords.x, grid.coords.y
        
        grid_corners = np.array([
            [x, y],
            [x + grid_size, y],
            [x + grid_size, y + grid_size],
            [x, y + grid_size]
        ], np.int32)
        
        cv2.fillPoly(self.frame, [grid_corners], color)
    
    def __call__(self, frame: np.ndarray) -> bool | np.ndarray:
        """
        Process a single frame with path detection and visualization.
        
        Args:
            frame: Input frame
            model: YOLO model instance
        
        Returns:
            Processed frame with visualizations
        """
        self.frame = frame
        
        # Check for blurry frames
        if self._reject_blurry_frames(frame):
            return False
               
        # Get YOLO results
        results = self.model.predict(frame, conf=0.5, verbose=self.verbose)
        
        # Extract grid information
        self._extract_grid_information(results)
        
        # If no grids were found, return original frame
        if not self.grids:
            return frame
        
        # Calculate penalties for each grid
        self._calculate_penalties()
        
        # Create graph for pathfinding
        graph = self._create_graph()
        
        # Detect protrusions
        protrusion_peaks = self.protrusion_detector(frame, self.grids)
        
        # Find paths
        paths = self._find_paths(protrusion_peaks, graph)
        
        # Draw non-path grids
        self._draw_non_path_grids()
        
        # Use PathAnalyser to visualize paths
        self.frame = path_analyser(self.frame, paths)
        
        return self.frame