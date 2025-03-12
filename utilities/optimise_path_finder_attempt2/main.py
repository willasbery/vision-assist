import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import cv2
import numpy as np
from pathlib import Path
from config import grid_size
from models import Grid
from PenaltyCalculator import penalty_calculator
from utilities.optimise_path_finder_attempt2.PathFinder import PathFinder

# Load the path examples
from utilities.optimise_path_finder_attempt2.path_examples import examples


def visualise_path(frame: np.ndarray, path: tuple[list[Grid], float], grid_lookup):
    # Visualise the penalties
    for grid in grid_lookup.values():
        if grid.penalty is None:
            continue
        
        penalty_colour = penalty_calculator.get_penalty_colour(grid.penalty)
    
        frame[
            grid.centre.y - grid_size // 2: grid.centre.y + grid_size // 2, 
            grid.centre.x - grid_size // 2: grid.centre.x + grid_size // 2
        ] = penalty_colour
        
    
    # Visualise the path
    path, cost = path
    for grid in path:
        frame[
            grid.centre.y - grid_size // 2: grid.centre.y + grid_size // 2, 
            grid.centre.x - grid_size // 2: grid.centre.x + grid_size // 2
        ] = (255, 0, 0)
        
    # Scale the image down in size by 25%
    frame = cv2.resize(frame, (3 * (frame.shape[1] // 4), 3 * (frame.shape[0] // 4)))
    
    return frame
    

def main(save_dir):
    # Create a new PathFinder instance
    # frame_processor = FrameProcessor()
    path_finder = PathFinder()
    
    example_number = 0

    for example in examples:
        graph = example["graph"]
        start_grid = example["start_grid"]
        end_grid = example["end_grid"]
        grid_lookup = example["grid_lookup"]
        
        highest_x = max([x + grid_size for _, x in grid_lookup.keys()])
        highest_y = max([y + grid_size for y, _ in grid_lookup.keys()])
        
        frame = np.zeros((highest_x, highest_y, 3), np.uint8)
        
        paths = path_finder.find_path(
            graph=graph,
            start_grid=start_grid,
            end_grid=end_grid,
            grid_lookup=grid_lookup
        )   
        
        frame = visualise_path(frame, paths, grid_lookup) 
        
        cv2.imwrite(save_dir / f"{example_number}.png", frame)
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        example_number += 1
    
    

if __name__ == "__main__":
    # ASKING FOR INPUT SO I DON'T ACCIDENTALLY RUN THE SAME COMMAND AND LOSE THE IMAGES
    base_dir = Path("./utilities/optimise_path_finder_attempt2/images")
    images_dir = input("Where should the images be saved? ")
    
    save_dir = base_dir / images_dir
    
    if not save_dir.exists():
        print(f"Directory {save_dir} does not exist, creating it now...")
        save_dir.mkdir(parents=True, exist_ok=True)
    
    main(save_dir)

