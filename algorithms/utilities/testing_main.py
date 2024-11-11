import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Import your existing modules
from models import Coordinate, Grid
from FrameProcessor import FrameProcessor
from PathAnalyser import path_analyser


def convert_npy_to_grid_info(npy_path: str, grid_size: int = 20) -> tuple[list[list[Grid]], dict[tuple[int, int], Grid]]:
    """
    Convert a numpy array file containing grid data to a format similar to extract_grid_information output.
    """
    grid_filled = np.load(npy_path)
    
    grids: list[list[Grid]] = []
    grid_lookup: dict[tuple[int, int], Grid] = {}
    
    for row_idx in range(grid_filled.shape[0]):
        this_row = []
        for col_idx in range(grid_filled.shape[1]):
            x = col_idx * grid_size
            y = row_idx * grid_size
            
            grid_centre = Coordinate(
                x=(x + grid_size // 2),
                y=(y + grid_size // 2)
            )
            
            grid = Grid(
                coords=Coordinate(x=x, y=y),
                centre=grid_centre,
                penalty=None,
                row=row_idx,
                col=col_idx,
                empty=not grid_filled[row_idx, col_idx]
            )
            
            this_row.append(grid)
            grid_lookup[(x, y)] = grid
            
        grids.append(this_row)
    
    return grids, grid_lookup


class EnhancedFrameProcessor(FrameProcessor):
    """
    Enhanced version of FrameProcessor that can work with both live detection and pre-saved grid data
    """
    def __init__(self, model: YOLO, verbose: bool) -> None:
        super().__init__(model, verbose)
        self.using_saved_grids = False
        
    def load_saved_data(self, frame: np.ndarray, grid_path: Path) -> None:
        """
        Load pre-saved grid data instead of running detection
        """
        self.frame = frame
        self.grids, self.grid_lookup = convert_npy_to_grid_info(str(grid_path))
        self.using_saved_grids = True
        
    def __call__(self, frame: np.ndarray) -> bool | np.ndarray:
        """
        Override the call method to handle both live detection and pre-saved data
        """
        self.frame = frame
        
        if not self.using_saved_grids:
            # Use normal processing path with YOLO detection
            return super().__call__(frame)
        
        # If using saved grids, skip detection and go straight to processing
        if not self.grids:
            return frame
            
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
        
        # Draw non-path grids
        self._draw_non_path_grids()
        
        # Use PathAnalyser to visualize paths
        self.frame = path_analyser(self.frame, paths)
        
        return self.frame

def main():
    # Initialize the enhanced processor
    frame_processor = EnhancedFrameProcessor(model="", verbose=False)
    
    # Load the frame and grid data
    frame_input = Path("./utilities/examples/segmented_grid_no_gridlines.png")
    grids = Path("./utilities/examples/grid_data.npy")
    
    frame = cv2.imread(str(frame_input))
    if frame is None:
        print(f"Error: Could not load frame image from {frame_input}")
        exit(1)
    
    # Load the saved grid data
    frame_processor.load_saved_data(frame, grids)
    
    # Process the frame
    processed_frame = frame_processor(frame)
    
    if isinstance(processed_frame, np.ndarray):
        # Display or save the processed frame
        cv2.imshow("Processed Frame", processed_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Frame processing failed")

if __name__ == "__main__":
    main()