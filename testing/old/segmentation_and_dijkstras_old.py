import argparse
import cv2
import numpy as np
import random
import heapq
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

# Helper function for Dijkstra's algorithm
def dijkstra(graph, start, goal):
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

def main(weights='yolov8n-seg.pt', device='cpu', source='../../videos/longer.MP4', output='./results/'):
    model = YOLO(f"{weights}")
    
    cap = cv2.VideoCapture(source)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps, fourcc = int(cap.get(5)), cv2.VideoWriter_fourcc(*"mp4v")
    
    save_dir = Path(output)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{Path(source).stem}.mp4"
    
    if save_path.exists():
        Path.unlink(save_path)
        
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()   
        
        if not ret:
            break
         
        results = model.predict(frame, conf=0.5)
    
        start_time = cv2.getTickCount()
        
        all_grids = []  # Store all grids
        grid_centres = {}  # Store the centres of each grid
        grid_size = 20  # Size of each grid
        frame_height, frame_width = frame.shape[:2]  # Get frame dimensions
        
        for result in results:
            if result.masks is None:
                continue
            
            for mask, box in zip(result.masks.xy, result.boxes):
                points = np.int32([mask])
                
                x, y, w, h = cv2.boundingRect(points)
                
                for i in range(x, x + w, grid_size):
                    for j in range(y, y + h, grid_size):
                        grid_centre = (i + grid_size // 2, j + grid_size // 2)
                        
                        # Check if the grid centre is inside the mask
                        if cv2.pointPolygonTest(points, grid_centre, False) >= 0:
                            grid = (i, j)  # Use the grid's top-left corner as its identifier
                            all_grids.append(grid)
                            grid_centres[grid] = grid_centre
                            
                            # Draw the grid rectangle
                            grid_corners = np.array([[i, j], [i + grid_size, j], [i + grid_size, j + grid_size], [i, j + grid_size]], np.int32)
                            cv2.polylines(frame, [grid_corners], isClosed=True, color=random.choices(range(256), k=3), thickness=1)
        
        # Create graph connections (edges between neighbouring grids)
        if len(all_grids) > 0:
            graph = defaultdict(list)
            for grid in all_grids:
                x, y = grid
                neighbours = [(x + grid_size, y), (x - grid_size, y), (x, y + grid_size), (x, y - grid_size)]
                
                for neighbour in neighbours:
                    if neighbour in grid_centres:
                        dist = np.linalg.norm(np.array(grid_centres[grid]) - np.array(grid_centres[neighbour]))
                        graph[grid].append((neighbour, dist))

            # Logic to pick start_grid and end_grid
            centre_x = frame_width // 2

            # Start grid: highest y-value and closest to the horizontal centre
            start_grid = max(all_grids, key=lambda grid: (grid[1], -abs(grid[0] - centre_x)))

            # End grid: lowest y-value and closest to the horizontal centre
            end_grid = min(all_grids, key=lambda grid: (grid[1], abs(grid[0] - centre_x)))

            print(f"Start grid: {start_grid}, End grid: {end_grid}")
            
            # Apply Dijkstra's algorithm to find the shortest path
            shortest_path, shortest_distance = dijkstra(graph, start_grid, end_grid)
            
            print(f"Shortest path: {shortest_path}")
            print(f"Shortest distance: {shortest_distance}")
            
            # Highlight the path on the frame
            for grid in shortest_path:
                i, j = grid
                grid_corners = np.array([[i, j], [i + grid_size, j], [i + grid_size, j + grid_size], [i, j + grid_size]], np.int32)
                cv2.polylines(frame, [grid_corners], isClosed=True, color=(0, 0, 255), thickness=2)  # Draw path in red
            
        end_time = cv2.getTickCount()
        print(f"Processing time: {(end_time - start_time) / cv2.getTickFrequency()} seconds")
    
        video_writer.write(frame)
    
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n-seg.pt', help='path to weights file')
    parser.add_argument('--device', type=str, default='cpu', help='device')
    parser.add_argument('--source', type=str, default='videos/longer.MP4', help='video file path')
    parser.add_argument('--output', type=str, default="./results/", help='output directory')
    
    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))