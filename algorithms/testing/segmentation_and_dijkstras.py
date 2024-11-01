import argparse
import cv2
import numpy as np
import heapq
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict


def dijkstra(graph: dict[list], start: tuple[int, int], goal: tuple[int, int]) -> tuple[list, float]:
    """
    Dijkstra's algorithm to find the shortest path between two nodes in a graph.
    
    :param graph: Graph represented as an adjacency list
    :param start: Starting node
    :param goal: Goal node
    
    :return: Tuple containing the shortest path and the distance between the start and goal nodes    
    """
    queue = [(0, start)]
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    previous_nodes = {node: None for node in graph}
    
    while queue:
        current_distance, current_node = heapq.heappop(queue)
        
        if current_node == goal:
            break
        
        for neighbour, weight in graph[current_node]:
            distance = current_distance + weight
            
            if distance < distances[neighbour]:
                distances[neighbour] = distance
                previous_nodes[neighbour] = current_node
                heapq.heappush(queue, (distance, neighbour))
    
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = previous_nodes[current]
    path.reverse()
    
    return path, distances[goal]


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
        
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()   
        if not ret:
            break
        
        results = model.predict(frame, conf=0.5)
    
        start_time = cv2.getTickCount()
        
        all_grids = []
        grid_centres = {}
        penalties = []
        grid_size = 20
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
                        # NOTE: I could use 20 here but this is not guaranteed to be the distance between the grids
                        dist = np.linalg.norm(np.array(grid_centres[grid]) - np.array(grid_centres[neighbour]))
                        graph[grid].append((neighbour, dist))

            # We want to favour grids that are closer to the centre and have lower penalties 
            # this assumes we are walking in a straight line from the start to the end, will
            # tackle corners later
            centre_x = frame_width // 2
            # Start grid: highest y-value and closest to the horizontal centre
            start_grid = max(all_grids, key=lambda grid: (grid[1], -abs(grid[0] - centre_x)))
            # End grid: lowest y-value and closest to the horizontal centre
            end_grid = min(all_grids, key=lambda grid: (grid[1], abs(grid[0] - centre_x)))
            
            shortest_path, shortest_distance = dijkstra(graph, start_grid, end_grid)
            
            # print(f"Start grid: {start_grid}, End grid: {end_grid}")
            # print(f"Shortest path: {shortest_path}")
            # print(f"Shortest path length: {shortest_distance}")
            
            for grid, penalty in zip(all_grids, penalties):
                i, j = grid
                grid_corners = np.array([
                    [i, j], # top left
                    [i + grid_size, j], # top right
                    [i + grid_size, j + grid_size], # bottom right
                    [i, j + grid_size] # bottom left
                ], np.int32)
                
                # If the grid is in the shortest path, draw its border in white, else colour it in red
                if grid in shortest_path:
                    cv2.fillPoly(frame, [grid_corners], [255, 255, 255])
                else:
                    cv2.fillPoly(frame, [grid_corners], [0, 0, 255])
            
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
