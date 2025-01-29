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


def calculate_path_direction(path: list[tuple[int, int]]) -> tuple:
    """
    Calculate the general direction of the path based on the slope and length of the path.
    
    :param path: List of grid coordinates in the path
    
    :return: tuple containing the bottom and top centre points of the path, the slope and the length of the path
    """
    bottom_grid = max(path, key=lambda g: g[1])  # Grid with max y (bottom)
    top_grid = min(path, key=lambda g: g[1])     # Grid with min y (top)
    
    bottom_x = bottom_grid[0]
    top_x = top_grid[0]
    
    centre_bottom = (bottom_x + (grid_size // 2), bottom_grid[1] + (grid_size // 2))
    centre_top = (top_x + (grid_size // 2), top_grid[1] + (grid_size // 2))
    
    delta_x = centre_top[0] - centre_bottom[0]
    delta_y = centre_top[1] - centre_bottom[1]
    slope = delta_y / delta_x if delta_x != 0 else float('inf')
    length = np.hypot(delta_x, delta_y)
    
    return centre_bottom, centre_top, slope, length


def detect_corner(close_slope, mid_slope, far_slope):
    """
    Detect if there's a corner based on the slopes of the close, middle, and far segments.
    
    :param close_slope: Slope of the close segment
    :param mid_slope: Slope of the middle segment
    :param far_slope: Slope of the far segment
    :return: A tuple (corner_type, confidence) where corner_type is 'right', 'left', or None,
             and confidence is a value between 0 and 1
    """
    # Convert infinite slopes to a large number for easier comparison
    close_slope = 1000 if abs(close_slope) == float('inf') else close_slope
    mid_slope = 1000 if abs(mid_slope) == float('inf') else mid_slope
    far_slope = 1000 if abs(far_slope) == float('inf') else far_slope
    
    # Calculate the differences between slopes
    diff_close_mid = abs(close_slope - mid_slope)
    diff_mid_far = abs(mid_slope - far_slope)
    
    # Check for right corner (slopes increasing in absolute value)
    if abs(far_slope) > abs(mid_slope) > abs(close_slope):
        confidence = min(diff_close_mid, diff_mid_far) / max(abs(close_slope), abs(mid_slope), abs(far_slope))
        return 'left', confidence
    
    # Check for left corner (slopes decreasing in absolute value)
    elif abs(close_slope) > abs(mid_slope) > abs(far_slope):
        confidence = min(diff_close_mid, diff_mid_far) / max(abs(close_slope), abs(mid_slope), abs(far_slope))
        return 'right', confidence
    
    # No corner detected
    return None, 0


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
          
            # Split the grids in to, close, mid and far from the camera based on the y-coordinate
            min_y = min(shortest_path, key=lambda g: g[1])[1]
            max_y = max(shortest_path, key=lambda g: g[1])[1]
            y_range = abs(max_y - min_y)
            
            far_grids = [grid for grid in shortest_path if grid[1] <= min_y + (y_range * 0.1)]
            mid_grids = [grid for grid in shortest_path if grid[1] > min_y + (y_range * 0.1) and grid[1] <= min_y + (y_range * 0.4)]
            close_grids = [grid for grid in shortest_path if grid[1] >= min_y + (y_range * 0.4)]
            
            for grid, penalty in zip(all_grids, penalties_dict.values()):
                i, j = grid
                grid_corners = np.array([
                    [i, j], # top left
                    [i + grid_size, j], # top right
                    [i + grid_size, j + grid_size], # bottom right
                    [i, j + grid_size] # bottom left
                ], np.int32)
                
                if grid in shortest_path:
                    if grid in close_grids:
                        cv2.fillPoly(frame, [grid_corners], [255, 187, 111])
                    elif grid in mid_grids:
                        cv2.fillPoly(frame, [grid_corners], [255, 53, 0])
                    elif grid in far_grids:
                        cv2.fillPoly(frame, [grid_corners], [255, 0, 97])
                    else:
                        cv2.fillPoly(frame, [grid_corners], [255, 255, 255])
                else:
                    color = get_penalty_colour(penalty)
                    cv2.fillPoly(frame, [grid_corners], color)
                    
            if far_grids and mid_grids and close_grids:
                # Get the general direction of the path for each segment
                far_bottom, far_top, far_slope, far_length = calculate_path_direction(far_grids)
                mid_bottom, mid_top, mid_slope, mid_length = calculate_path_direction(mid_grids)
                close_bottom, close_top, close_slope, close_length = calculate_path_direction(close_grids)
                
                # Draw the path segments on the frame
                cv2.line(frame, far_bottom, far_top, [255, 255, 255], 2)
                cv2.line(frame, mid_bottom, mid_top, [255, 255, 255], 2)
                cv2.line(frame, close_bottom, close_top, [255, 255, 255], 2)
                
                # Add text to show the slope and length of each segment
                cv2.putText(frame, f"Far: {far_slope:.2f}, {far_length:.2f}", far_bottom, cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1, cv2.LINE_AA)
                cv2.putText(frame, f"Mid: {mid_slope:.2f}, {mid_length:.2f}", mid_bottom, cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1, cv2.LINE_AA)
                cv2.putText(frame, f"Close: {close_slope:.2f}, {close_length:.2f}", close_bottom, cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1, cv2.LINE_AA)
                
                corner_type, confidence = detect_corner(close_slope, mid_slope, far_slope)
                if corner_type:
                    cv2.putText(frame, f"{corner_type.capitalize()} corner detected: {confidence:.2f}", (30, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 2, cv2.LINE_AA)
                
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
