import cv2
import numpy as np
from collections import defaultdict
from typing import ClassVar, Optional
from ultralytics import YOLO

import time

from config import grid_size
from models import Coordinate, Grid, Instruction, Path
from PathAnalyser import path_analyser
from PathFinder_for_saving_path_examples import path_finder
from PathVisualiser import path_visualiser
from PenaltyCalculator import penalty_calculator
from ProtrusionDetector import ProtrusionDetector
from utils import get_closest_grid_to_point


class FrameProcessor:
    """
    FrameProcessor handles the processing of video frames for path detection.
    """
    _instance: ClassVar[Optional['FrameProcessor']] = None
    _initialized: bool = False
    
    def __new__(cls, model: YOLO, verbose: bool, debug: bool, timing_data_path: Path) -> 'FrameProcessor':  
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model: YOLO, verbose: bool, debug: bool, timing_data_path: Path) -> None:
        """Initialize the processor only once."""
        if not self._initialized:
            self._initialized = True
            self.model = model
            self.verbose = verbose
            self.debug = debug
            
            self.frame: Optional[np.ndarray] = None
            self.grids: list[list[Grid]] = [] # 2d array of grids
            self.grid_lookup: dict[tuple[int, int], Grid] = {} # (x, y) -> Grid mapping
            self.np_grids: np.ndarray = np.empty((0, 0), dtype=np.uint8)
            self.protrusion_detector = ProtrusionDetector(debug=debug)
            
            self.timing_data_path = timing_data_path
            self.timing_data: dict[str, list[int]] = defaultdict(list)  # Store timing information
            
    def _reject_blurry_frames(self, frame: np.ndarray) -> bool:
        """Reject blurry frames based on the Laplacian variance."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_level = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
        return blur_level < 100
    
    def _extract_grid_information(self, results) -> None:
        """Extract grid information from YOLO detection results."""
        self.grids.clear()
        self.grid_lookup.clear()
        self.np_grids = np.empty((0, 0), dtype=np.uint8)
        
        for result in results:
            if result.masks is None:
                continue
            
            mask_count = len(result.masks.xy)
            mask = max(result.masks.xy, key=lambda x: cv2.contourArea(x)) if mask_count > 1 else result.masks.xy[0]
                
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
            
            # Add the artificial grids to the np_grids array
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
                    col = (j - x) // grid_size
                    grid = Grid(
                        coords=Coordinate(x=j, y=i), 
                        centre=grid_centre, 
                        penalty=None, 
                        row=row_count, 
                        col=col, 
                        empty=empty,
                        artificial=artificial
                    ) 
                                                    
                    self.grid_lookup[(j, i)] = grid
                    this_row.append(grid)
                
                if row_count < len(self.grids) - 1:
                    self.grids[row_count] = this_row
                else:
                    self.grids.append(this_row)
                    
            # At the end, we need to create a numpy array of the grid
            number_of_rows = len(self.grids)
            number_of_columns = len(self.grids[0])
            self.np_grids = np.zeros((number_of_rows, number_of_columns), dtype=np.uint8)
            
            for row_index, row in enumerate(self.grids):
                for col_index, grid in enumerate(row):
                    self.np_grids[row_index, col_index] = 0 if grid.empty else 1
    
    def _calculate_penalties(self) -> None:
        """Calculate penalties for each grid."""
        penalty_calculator._pre_compute_easy_segments(self.np_grids, self.grids)
        
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
        start_blurry_frame_check = time.process_time_ns()
        if self._reject_blurry_frames(frame):
            if self.debug:
                return self.frame, []
            else:
                return []
        end_blurry_frame_check = time.process_time_ns()
               
        # Get YOLO results
        start_yolo_prediction = time.process_time_ns()
        results = self.model.predict(frame, conf=0.5, verbose=self.verbose)
        end_yolo_prediction = time.process_time_ns()
        
        if not results:
            if self.debug:
                return self.frame, []
            else:
                return []
        
        # Extract grid information
        start_grid_extraction = time.process_time_ns()
        self._extract_grid_information(results)
        end_grid_extraction = time.process_time_ns()
        
        # If no grids were found, return original frame
        if not self.grids:
            if self.debug:
                return self.frame, []
            else:
                return []
        
        # Calculate penalties for each grid
        start_penalty_calculation = time.process_time_ns()
        self._calculate_penalties()
        end_penalty_calculation = time.process_time_ns()
        
        # Create graph for pathfinding
        start_graph_creation = time.process_time_ns()
        graph = self._create_graph()
        end_graph_creation = time.process_time_ns()
        
        # Detect protrusions
        start_protrusion_detection = time.process_time_ns()
        protrusion_peaks = self.protrusion_detector(frame, self.grids, self.grid_lookup)
        end_protrusion_detection = time.process_time_ns()
        
        if not protrusion_peaks:
            print("No protrusions detected.")
        
        # Find paths
        start_path_finding = time.process_time_ns()
        paths = self._find_paths(protrusion_peaks, graph)
        end_path_finding = time.process_time_ns()
        
        # Analyse paths
        start_path_analysis = time.process_time_ns()
        final_answer = path_analyser(frame.shape[0], frame.shape[1], paths)
        end_path_analysis = time.process_time_ns()


        if end_yolo_prediction - start_yolo_prediction > 1_000_000_000:
            print(f"YOLO prediction took {(end_yolo_prediction - start_yolo_prediction) / 1_000_000_000} seconds")
            print("Skipping timing data because it ruins my graphs lmao")
        else:
            # Save timing data to file
            print(f"Adding timing data, instructions are {final_answer}")
            self.timing_data["blurry_frame_check"].append((end_blurry_frame_check - start_blurry_frame_check) / 1_000_000_000)
            self.timing_data["yolo_prediction"].append((end_yolo_prediction - start_yolo_prediction) / 1_000_000_000)
            self.timing_data["grid_extraction"].append((end_grid_extraction - start_grid_extraction) / 1_000_000_000)
            self.timing_data["penalty_calculation"].append((end_penalty_calculation - start_penalty_calculation) / 1_000_000_000)
            self.timing_data["graph_creation"].append((end_graph_creation - start_graph_creation) / 1_000_000_000)
            self.timing_data["protrusion_detection"].append((end_protrusion_detection - start_protrusion_detection) / 1_000_000_000)
            self.timing_data["path_finding"].append((end_path_finding - start_path_finding) / 1_000_000_000)
            self.timing_data["path_analysis"].append((end_path_analysis - start_path_analysis) / 1_000_000_000)
            
            self._save_timing_data()
        
        if self.debug:
            # Draw non-path grids
            self._draw_non_path_grids()
            
            # Use PathAnalyser to visualize paths
            self.frame = path_visualiser(self.frame, paths)
            
            return self.frame, final_answer
        else:
            return final_answer

    def _save_timing_data(self) -> None:
        """Save timing data to a text file with averages."""
        with open(f'{self.timing_data_path}/timing_data.txt', 'w') as f:
            f.write("Timing Data (nanoseconds)\n")
            f.write("=======================\n\n")
            
            for operation, times in self.timing_data.items():
                avg_time = sum(times) / len(times) if times else 0
                f.write(f"{operation}:\n")
                f.write(f"  Average: {avg_time:.2f}\n")
                f.write(f"  Last: {times[-1] if times else 0}\n")
                f.write(f"  Min: {min(times) if times else 0}\n")
                f.write(f"  Max: {max(times) if times else 0}\n\n")