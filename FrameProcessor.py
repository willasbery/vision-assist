import cv2
import numpy as np
from collections import defaultdict
from typing import ClassVar, Optional
from ultralytics import YOLO

from vision_assist.config import grid_size
from vision_assist.models import Coordinate, Grid, Instruction, Path
from vision_assist.PathAnalyser import path_analyser
from vision_assist.PathFinder import path_finder
from vision_assist.PathVisualiser import path_visualiser
from vision_assist.PenaltyCalculator import penalty_calculator
from vision_assist.ProtrusionDetector import ProtrusionDetector
from vision_assist.utils import get_closest_grid_to_point


class FrameProcessor:
    """
    FrameProcessor handles the processing of video frames for path detection.
    """
    _instance: ClassVar[Optional['FrameProcessor']] = None
    _initialized: bool = False
    
    def __new__(cls, model: YOLO, verbose: bool, debug: bool, imshow: bool) -> 'FrameProcessor':  
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model: YOLO, verbose: bool, debug: bool, imshow: bool) -> None:
        """Initialize the processor only once."""
        if not self._initialized:
            self._initialized = True
            self.model = model
            self.verbose = verbose
            self.debug = debug
            self.imshow = imshow
            
            self.frame: Optional[np.ndarray] = None
            self.grids: list[list[Grid]] = [] # 2d array of grids
            self.grid_lookup: dict[tuple[int, int], Grid] = {} # (x, y) -> Grid mapping
            self.protrusion_detector = ProtrusionDetector(debug=debug, imshow=imshow)
            
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
                w = self.frame.shape[1] if w > self.frame.shape[1] else w                
                h = h + (grid_size - h % grid_size) if h % grid_size != 0 else h
                
                artifical_grid_column_xs = [
                    x for x in range(
                        (self.frame.shape[1] // 2) - (grid_size * 8), 
                        (self.frame.shape[1] // 2) + (grid_size * (8 + 1)), grid_size
                    )
                ]
                added_a_grid = False
                    
                # First create grids for the actual mask
                row_count = 0
                for i in range(y, y + h, grid_size):
                    col_count = 0
                    this_row = []
                    
                    for j in range(x, x + w, grid_size):
                        grid_centre = Coordinate(x=(j + grid_size // 2), y=(i + grid_size // 2))
                        in_mask = cv2.pointPolygonTest(points, grid_centre.to_tuple(), False) >= 0
                        
                        if in_mask: added_a_grid = True
                        
                        grid = Grid(
                            coords=Coordinate(x=j, y=i), 
                            centre=grid_centre, 
                            penalty=None, 
                            row=row_count, 
                            col=col_count, 
                            empty=not in_mask,
                            artificial=False
                        )
                                                
                        this_row.append(grid)
                        self.grid_lookup[(j, i)] = grid
                        col_count += 1
                    
                    self.grids.append(this_row)
                    row_count += 1
                    
                if not added_a_grid:
                    print("No grids were added for the mask.")
                    continue
                         
                starting_y = int(self.frame.shape[0] * 0.875) + (grid_size - int(self.frame.shape[0] * 0.875) % grid_size)
                
                for i in range (starting_y, self.frame.shape[0], grid_size):
                    row_count = (i - y) // grid_size
                    this_row = []
                 
                    for j in range(x, x + w, grid_size):
                        previously_empty = self.grid_lookup.get((j, i)).empty if self.grid_lookup.get((j, i)) else True
                                    
                        if previously_empty:
                            if j in artifical_grid_column_xs:
                                empty = False
                                artificial = True
                            else:
                                empty = True
                                artificial = False
                        else:
                            empty = False
                            artificial = False
                                
                        grid_centre = Coordinate(x=(j + grid_size // 2), y=(i + grid_size // 2))   
                        grid = Grid(
                            coords=Coordinate(x=j, y=i), 
                            centre=grid_centre, 
                            penalty=None, 
                            row=row_count, 
                            col=(j - x) // grid_size, 
                            empty=empty,
                            artificial=artificial
                        ) 
                                                     
                        self.grid_lookup[(j, i)] = grid
                        this_row.append(grid)
                    
                    if row_count < len(self.grids) - 1:
                        self.grids[row_count] = this_row
                    else:
                        self.grids.append(this_row)
    
    def _calculate_penalties(self) -> None:
        """Calculate penalties for each grid."""
        for grid_row in self.grids:
            for grid in grid_row:
                if grid.empty:
                    continue
                
                grid.penalty = penalty_calculator.calculate_penalty(grid, self.grid_lookup)
    
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
    
    def _calculate_path_similarity(self, path1: Path, path2: Path) -> float:
        """
        Calculate path similarity using section-based comparison.
        """
        grids1 = {(g.coords.x, g.coords.y) for g in path1.grids}
        grids2 = {(g.coords.x, g.coords.y) for g in path2.grids}
        
        if not grids1 or not grids2:
            return 0.0
        
        intersection = len(grids1.intersection(grids2))
        
        # Check if one path is a subset of the other
        if intersection == len(grids1) or intersection == len(grids2):
            return 1.0
        
        union = len(grids1.union(grids2))
        
        # Jaccard similarity coefficient
        return intersection / union if union > 0 else 0.0
        
    def _find_paths(self, protrusion_peaks: list[Coordinate], graph: defaultdict) -> list[Path]:
        """Find paths using PathFinder with enhanced path analysis."""
        all_paths = []
        if not self.grids:
            return all_paths
            
        # Always start from the middle of the last row
        start_grid = get_closest_grid_to_point(Coordinate(x=self.frame.shape[1] // 2, y=self.frame.shape[0]), self.grids)
        
        for peak in protrusion_peaks:
            closest_grid = get_closest_grid_to_point(peak, self.grids)
            
            grid_path, total_cost = path_finder.find_path(graph, start_grid, closest_grid, self.grid_lookup)
            
            if grid_path:
                path = Path(
                    grids=grid_path,
                    total_cost=total_cost, 
                    path_type="path"
                )
                all_paths.append(path)
            else:
                print("No path found.")
        
        # Filter similar paths using the new section-based similarity
        unique_paths: list[Path] = []
        all_paths.sort(key=lambda x: len(x.grids), reverse=True)
        
        for path in all_paths:
            is_unique = True
            
            for existing_path in unique_paths:
                similarity = self._calculate_path_similarity(path, existing_path)
                
                if similarity >= 0.90:  # Adjusted threshold for section-based comparison
                    is_unique = False
                    break
            
            if is_unique:
                unique_paths.append(path)
        
        return unique_paths
    
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
    
    def _draw_grid(self, grid: dict, color: tuple[int, int, int]) -> None:
        """Draw a single grid on the frame."""
        x, y = grid.coords.x, grid.coords.y
        
        grid_corners = np.array([
            [x, y],
            [x + grid_size, y],
            [x + grid_size, y + grid_size],
            [x, y + grid_size]
        ], np.int32)
        
        cv2.fillPoly(self.frame, [grid_corners], color)
    
    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, list[Instruction]] | list[Instruction]:
        """
        Process a single frame with path detection and visualization.
        
        Args:
            frame: Input frame
            model: YOLO model instance
        
        Returns:
            Whatever it wants, whenever it wants, to whoever it wants
        """
        self.frame = frame
        
        # Check for blurry frames
        # if self._reject_blurry_frames(frame):
        #     if self.debug:
        #         return self.frame, []
        #     else:
        #         return []
               
        # Get YOLO results
        results = self.model.predict(frame, conf=0.5, verbose=self.verbose)
        
        # Extract grid information
        self._extract_grid_information(results)
        
        # If no grids were found, return original frame
        if not self.grids:
            if self.debug:
                return self.frame, []
            else:
                return []
        
        # Calculate penalties for each grid
        self._calculate_penalties()
        
        # Create graph for pathfinding
        graph = self._create_graph()
        
        # Detect protrusions
        protrusion_peaks = self.protrusion_detector(frame, self.grids, self.grid_lookup)
        
        if not protrusion_peaks:
            print("No protrusions detected.")
        
        # Find paths
        paths = self._find_paths(protrusion_peaks, graph)
        
        instructions = path_analyser(frame.shape[0], paths)
  
        if self.debug:
            # Draw non-path grids
            self._draw_non_path_grids()
            
            # Use PathAnalyser to visualize paths
            self.frame = path_visualiser(self.frame, paths)
            
            return self.frame, instructions
        else:
            return instructions