import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Import your existing modules
from models import Coordinate, Grid
from FrameProcessor import FrameProcessor
from PathAnalyser import path_analyser

def setup_argparse() -> argparse.Namespace:
    """
    Set up command line argument parsing.
    """
    parser = argparse.ArgumentParser(description='Process grid data and visualize paths.')
    parser.add_argument('base_filename', type=str, 
                       help='Base filename without extensions (e.g., "my-drawing" for "my-drawing_img.png" and "my-drawing_grids.npy")')
    parser.add_argument('--input-dir', type=str, default='./examples',
                       help='Directory containing input files (default: ./examples)')
    return parser.parse_args()

def get_file_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """
    Construct file paths from base filename and directories.
    Returns paths for input image, grid data, and output image.
    """
    # Ensure directories exist
    
    # Construct paths
    image_path = Path(args.input_dir) / f"{args.base_filename}_img.png"
    grid_path = Path(args.input_dir) / f"{args.base_filename}_grids.npy"
    
    # Verify input files exist
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid data file not found: {grid_path}")
    
    return image_path, grid_path

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

class SingleSavedFrameFrameProcessor(FrameProcessor):
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
    # Parse command line arguments
    args = setup_argparse()
    
    try:
        # Get file paths
        image_path, grid_path = get_file_paths(args)
        
        # Initialize the enhanced processor
        frame_processor = SingleSavedFrameFrameProcessor(model="", verbose=False)
        
        # Load the frame
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Could not load frame image from {image_path}")
        
        # Load the saved grid data
        frame_processor.load_saved_data(frame, grid_path)
        
        # Process the frame
        processed_frame = frame_processor(frame)
        
        if isinstance(processed_frame, np.ndarray):
            # Display the processed frame
            cv2.imshow("Processed Frame", processed_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Frame processing failed")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()