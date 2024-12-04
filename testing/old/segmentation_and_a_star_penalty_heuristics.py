import argparse
import cv2
import numpy as np
import heapq
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict


grid_size = 20
# stored as BGR!
penalty_colour_gradient = {
    1.0000: (0, 0, 255),
    0.9166: (0, 60, 255),
    0.8333: (0, 88, 255),
    0.7500: (0, 109, 255),
    0.6666: (0, 128, 255),
    0.5833: (8, 145, 255),
    0.5000: (0, 163, 249), 
    0.4166: (0, 183, 232),
    0.3333: (0, 202, 208),
    0.1666: (0, 221, 176),
    0.0833: (0, 239, 129),
    0.0000: (0, 255, 15)
}


def a_star(graph: dict[list], start: tuple[int, int], goal: tuple[int, int], penalties: dict) -> tuple[list, float]:
    """
    A* algorithm to find the shortest path between two nodes in a graph, with the heuristic being the penalty for each grid.
    
    :param graph: Graph represented as an adjacency list
    :param start: Starting node
    :param goal: Goal node
    :param penalties: A dictionary of penalties for each grid (heuristic)
    
    :return: Tuple containing the shortest path and the distance between the start and goal nodes    
    """
    queue = [(0, start)]
    all_nodes = penalties.keys()
    g_scores = {node: float('inf') for node in all_nodes}
    g_scores[start] = 0
    f_scores = {node: float('inf') for node in all_nodes}
    f_scores[start] = penalties[start]
    previous_nodes = {node: None for node in all_nodes}
    
    while queue:
        current_f_score, current_node = heapq.heappop(queue)
        
        if current_node == goal:
            break
        
        for neighbour, weight in graph.get(current_node, []):
            tentative_g_score = g_scores[current_node] + weight
            
            if tentative_g_score < g_scores[neighbour]:
                g_scores[neighbour] = tentative_g_score
                
                # f = g + h (heuristic)
                f_score = tentative_g_score + ((tentative_g_score * penalties[neighbour]) ** 3)
                 
                f_scores[neighbour] = f_score
                previous_nodes[neighbour] = current_node
                heapq.heappush(queue, (f_score, neighbour))
    
    # Reconstruct the path
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = previous_nodes[current]
    path.reverse()
    
    return path, g_scores[goal]


def get_penalty_colour(penalty: float) -> tuple:
    """ 
    Get the colour based on the penalty value. 
    
    :param penalty: Penalty value between 0 and 1
    :return: BGR colour tuple
    """
    closest_key = min(penalty_colour_gradient.keys(), key=lambda x: abs(x - penalty))
    return penalty_colour_gradient[closest_key]


def calculate_row_penalty(x: int, min_x: int, max_x: int) -> float:
    """ 
    Calculate the penalty for a grid based on its position within a row. 
    
    :param x: x-coordinate of the grid
    :param min_x: x-coordinate of the leftmost grid in the row
    :param max_x: x-coordinate of the rightmost grid in the row
    
    :return: Penalty value for the grid between 0 and 1
    """
    row_width = max_x - min_x
    if row_width == 0:
        return 0  # If there's only one grid in the row, no penalty
    
    # Normalise the position within the row (0 to 1)
    position_ratio = (x - min_x) / row_width
    
    # Calculate penalty based on the position within the row, with the highest penalty at the edges
    # Distribution:
    # 1      0.5        0        0.5        1
    # . . . . . . . . . . . . . . . . . . . .
    penalty = 2 * abs(position_ratio - 0.5)
    return penalty


def main(
        weights: str | Path ='yolov8n-seg.pt', 
        source: str | Path ='../../videos/longer.MP4', 
        output: str | Path ='./results/'
    ) -> None:
    """ 
    Main function to process the video and generate the output.
    
    :param weights: Path to the weights file
    :param source: Path to the video file
    :param output: Output directory
    
    :return: None, but saves the processed video to the output directory
    """
    model = YOLO(f"{weights}")
    
    cap = cv2.VideoCapture(source)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps, fourcc = int(cap.get(5)), cv2.VideoWriter_fourcc(*"mp4v")
    
    save_dir = Path(output)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{Path(source).stem}_experimental.mp4"
    
    # Delete video if it already exists as cv2.VideoWriter does not overwrite
    if save_path.exists():
        Path.unlink(save_path)
        
    video_writer = cv2.VideoWriter(str(save_path), fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()   
        if not ret:
            break
        
        results = model.predict(frame, conf=0.5)
    
        start_time = cv2.getTickCount()
        
        all_grids = []
        grid_centres = {}
        penalties_dict = {}
        row_dict = defaultdict(list)  # Store grids by rows (y-coordinate)
        frame_height, frame_width = frame.shape[:2]
        
        for result in results:
            if result.masks is None:
                continue
            
            for mask in result.masks.xy:
                points = np.int32([mask])
                
                x, y, w, h = cv2.boundingRect(points)
                
                for i in range(y, y + h, grid_size):
                    for j in range(x, x + w, grid_size):
                        grid_centre = (j + grid_size // 2, i + grid_size // 2)
                        
                        # Check if the grid centre is inside the mask
                        if cv2.pointPolygonTest(points, grid_centre, False) >= 0:
                            grid = (j, i)  # Use the grid's top-left corner as its identifier
                            all_grids.append(grid)
                            grid_centres[grid] = grid_centre
                            row_dict[grid_centre[1]].append(grid)  # Add grid to the row (group by y-coordinate)
        
        # Calculate the penalty for each grid based on its distance from the edges of the row
        for _, grids_in_row in row_dict.items():
            min_x = min(grids_in_row, key=lambda g: g[0])[0]  # Leftmost grid in the row
            max_x = max(grids_in_row, key=lambda g: g[0])[0]  # Rightmost grid in the row
            
            for grid in grids_in_row:
                x, _ = grid
                penalty = calculate_row_penalty(x, min_x, max_x)
                penalties_dict[grid] = penalty
        
        # Create graph connections (edges between neighbouring grids)
        if len(all_grids) > 0:
            graph = defaultdict(list)
            for grid in all_grids:
                x, y = grid
                neighbours = [
                    (x + grid_size, y), # Right
                    (x - grid_size, y), # Left
                    (x, y + grid_size), # Down
                    (x, y - grid_size)  # Up
                ]
                
                for neighbour in neighbours:
                    if neighbour in grid_centres:
                        # Calculate the Euclidean distance between the grid centres
                        dist = np.linalg.norm(np.array(grid_centres[grid]) - np.array(grid_centres[neighbour]))
                        graph[grid].append((neighbour, dist))

            # Start grid: highest y-value and closest to the horizontal centre of the bottom row
            bottom_row_grids = [grid for grid in all_grids if grid[1] == max(all_grids, key=lambda g: g[1])[1]]
            bottom_row_x_values = [grid[0] for grid in bottom_row_grids]
            bottom_row_centre_x = sum(bottom_row_x_values) // len(bottom_row_x_values)  # Middle x-coordinate of the bottom row
            start_grid = min(bottom_row_grids, key=lambda grid: abs(grid[0] - bottom_row_centre_x))
            
            # End grid: lowest y-value and closest to the horizontal centre
            centre_x = frame_width // 2
            end_grid = min(all_grids, key=lambda grid: (grid[1], abs(grid[0] - centre_x)))
            
            shortest_path, shortest_distance = a_star(graph, start_grid, end_grid, penalties_dict)
            
            for grid, penalty in zip(all_grids, penalties_dict.values()):
                i, j = grid
                grid_corners = np.array([
                    [i, j], # top left
                    [i + grid_size, j], # top right
                    [i + grid_size, j + grid_size], # bottom right
                    [i, j + grid_size] # bottom left
                ], np.int32)
                
                # If the grid is in the shortest path, draw its border in white, else colour it based on the penalty
                if grid in shortest_path:
                    cv2.fillPoly(frame, [grid_corners], [255, 255, 255])
                else:
                    color = get_penalty_colour(penalty)
                    cv2.fillPoly(frame, [grid_corners], color)
            
        end_time = cv2.getTickCount()
        print(f"Processing time: {(end_time - start_time) / cv2.getTickFrequency()} seconds")
    
        video_writer.write(frame)
    
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n-seg.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='videos/longer.MP4', help='video file path')
    parser.add_argument('--output', type=str, default="./results/", help='output directory')
    
    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))
