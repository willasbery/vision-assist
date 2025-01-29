import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import cv2
import numpy as np
from pathlib import Path as PathLibPath
from ultralytics import YOLO

# Import your existing modules
from models import Coordinate, Grid, Path
from FrameProcessor import FrameProcessor
from PathVisualiser import path_visualiser


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

def get_file_paths(args: argparse.Namespace) -> tuple[PathLibPath, PathLibPath]:
    """
    Construct file paths from base filename and directories.
    Returns paths for input image and grid data.
    """
    # Construct paths
    image_path = PathLibPath(args.input_dir) / f"{args.base_filename}_img.png"
    grid_path = PathLibPath(args.input_dir) / f"{args.base_filename}_grids.npy"
    
    # Verify input files exist
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid data file not found: {grid_path}")
    
    return image_path, grid_path

def convert_npy_to_grid_info(npy_path: str, grid_size: int = 20) -> tuple[list[list[Grid]], dict[tuple[int, int], Grid]]:
    """
    Convert a numpy array file containing grid data to a format similar to extract_grid_information output.
    Using Pydantic v2 initialization syntax and including artificial grid columns.
    """
    grid_filled = np.load(npy_path)
    frame_height, frame_width = grid_filled.shape
    frame_height *= grid_size
    frame_width *= grid_size
    
    print(f"Frame dimensions: {frame_width}x{frame_height}")
    
    grids: list[list[Grid]] = []
    grid_lookup: dict[tuple[int, int], Grid] = {}
    
    # Calculate artificial grid columns
    artifical_grid_column_xs = [
        x for x in range(
            (frame_width // 2) - (grid_size * 8), 
            (frame_width // 2) + (grid_size * (8 + 1)), 
            grid_size
        )
    ]
    
    print(f"Artificial grid columns: {artifical_grid_column_xs}")
    
    # First create grids for the main area
    row_count = 0
    for row_idx in range(grid_filled.shape[0]):
        this_row = []
        col_count = 0
        
        for col_idx in range(grid_filled.shape[1]):
            grid_x = col_idx * grid_size
            grid_y = row_idx * grid_size
            
            coords = Coordinate(x=grid_x, y=grid_y)
            grid_centre = Coordinate(x=(grid_x + grid_size // 2), y=(grid_y + grid_size // 2))
            
            grid_filled_here = grid_filled[row_idx, col_idx]
            
            grid = Grid(
                coords=coords,
                centre=grid_centre,
                penalty=None,
                row=row_count,
                col=col_count,
                empty=not grid_filled_here,
                artificial=False
            )
            
            this_row.append(grid)
            grid_lookup[(grid_x, grid_y)] = grid
            col_count += 1
        
        grids.append(this_row)
        row_count += 1
        
    
    starting_y = int(frame_height * 0.8375) + (grid_size - int(frame_height * 0.8375) % grid_size)
    
    for i in range(starting_y, frame_height, grid_size):
        row_count = i // grid_size
        this_row = []
        
        for j in range(0, frame_width, grid_size):
            previously_empty = grid_lookup.get((j, i)).empty if grid_lookup.get((j, i)) else True
            
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
            coords = Coordinate(x=j, y=i)
            
            grid = Grid(
                coords=coords,
                centre=grid_centre,
                penalty=None,
                row=row_count,
                col=j // grid_size,
                empty=empty,
                artificial=artificial
            )
            
            grid_lookup[(j, i)] = grid
            this_row.append(grid)
        
        if row_count < len(grids):
            grids[row_count] = this_row
        else:
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
        
        cv2.imwrite(f"generate_testing_grids/examples/outputs/grids_visualised_2_original.png", frame)
        cv2.imshow("Original Frame", frame)
            
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
        
        # Draw non-path grids with penalties
        self._draw_non_path_grids()
        
        # Use PathAnalyser to visualize paths
        # self.frame = path_visualiser(self.frame, paths)
        
        return self.frame

def main():
    # Parse command line arguments
    args = setup_argparse()
    
    try:
        # Get file paths
        image_path, grid_path = get_file_paths(args)
        
        # Initialize the enhanced processor
        frame_processor = SingleSavedFrameFrameProcessor(model=None, verbose=False)
        
        # Load the frame
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Could not load frame image from {image_path}")
        
        # Load the saved grid data
        frame_processor.load_saved_data(frame, grid_path)
        
        # Process the frame
        processed_frame = frame_processor(frame)
        
        if isinstance(processed_frame, np.ndarray):
            cv2.imwrite(f"generate_testing_grids/examples/outputs/{args.base_filename}_processed.png", processed_frame)
            print("Saved processed frame to examples/outputs")
            processed_frame = cv2.resize(processed_frame, (576, 1024))
            # Display the processed frame
            cv2.imshow("Processed Frame", processed_frame)
            # cv2.imwrite(f"/outputs/{args.base_filename}_processed.png", processed_frame)
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