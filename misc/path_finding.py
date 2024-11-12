import numpy as np

from heapq import heappop, heappush

from config import grid_size, penalty_colour_gradient


def a_star(
    graph: dict, 
    start_grid: dict, 
    end_grid: dict, 
    grid_lookup: dict[tuple[int, int], dict]) -> tuple[list[dict], float]:
    """
    Implement A* pathfinding algorithm to find the optimal path between start and end grids.
    
    Args:
        graph: Dictionary containing grid connections and distances
        start_grid: Starting grid dictionary
        end_grid: Target grid dictionary
        grids: List of all grid dictionaries with penalties
    
    Returns:
        Tuple containing:
        - List of grid dictionaries representing the optimal path
        - Total cost of the path
    """    
  
    def heuristic(grid1: dict, grid2: dict) -> float:
        """Calculate the Manhattan distance between two grids."""
        return abs(grid1['coords']['x'] - grid2['coords']['x']) + \
               abs(grid1['coords']['y'] - grid2['coords']['y'])
    
    # Initialize sets and scores using coordinates as keys
    start_coord = (start_grid['coords']['x'], start_grid['coords']['y'])
    end_coord = (end_grid['coords']['x'], end_grid['coords']['y'])
    
    open_set = []
    closed_set = set()
    came_from = {}
    g_score = {start_coord: 0}
    f_score = {start_coord: heuristic(start_grid, end_grid)}
    
    # Add start grid to open set
    heappush(open_set, (f_score[start_coord], start_coord))
    
    while open_set:
        current_coords = heappop(open_set)[1]
        current_grid = grid_lookup[current_coords]
        
        # Check if we've reached the end
        if current_coords == end_coord:
            # Reconstruct path
            path = []
            current = current_coords
            total_cost = g_score[current]
            
            while current in came_from:
                path.append(grid_lookup[current])
                current = came_from[current]
            
            path.append(start_grid)
            path.reverse()
            return path, total_cost
        
        closed_set.add(current_coords)
        
        # Check all neighbors
        for neighbor_coords, distance in graph[current_coords]:
            if neighbor_coords in closed_set:
                continue
            
            neighbor_grid = grid_lookup[neighbor_coords]
            
            # Calculate tentative g_score with penalty
            penalty_multiplier = 1 + (neighbor_grid['penalty'] or 0)
            tentative_g_score = g_score[current_coords] + (distance * penalty_multiplier)
            
            # If this path is better than any previous one
            if neighbor_coords not in g_score or tentative_g_score < g_score[neighbor_coords]:
                came_from[neighbor_coords] = current_coords
                g_score[neighbor_coords] = tentative_g_score
                f_score[neighbor_coords] = tentative_g_score + heuristic(neighbor_grid, end_grid)
                
                # Add to open set if not already there
                if not any(coords == neighbor_coords for _, coords in open_set):
                    heappush(open_set, (f_score[neighbor_coords], neighbor_coords))
    
    # No path found
    return [], float('inf')


def calculate_row_penalty(current_grid: dict, grids: list[list[dict]]) -> float:
    """
    Calculate penalty for a grid based on its position within continuous segments of grids in a row.
    Higher penalties are assigned to grids further from the center of their segment.
    
    Args:
        current_grid: Dictionary containing grid information with 'row' and 'coords' keys
        grids: List of rows, where each row contains grid dictionaries
        grid_size: Size of each grid square (default=20)
    
    Returns:
        float: Penalty value between 0 and 1, where:
            0 = single grid or at segment center
            1 = at segment edge
    """
    row = current_grid['row']
    row_grids = grids[row]
    current_coords = current_grid['coords']
    
    # Find current grid index in the row
    left, right = 0, len(row_grids) - 1
    while left <= right:
        mid = (left + right) // 2
        if row_grids[mid]['coords'] == current_coords:
            current_idx = mid
            break
        elif row_grids[mid]['coords']['x'] < current_coords['x']:
            left = mid + 1
        else:
            right = mid - 1
    
    if len(row_grids) == 1:
        return 0
    
    x = current_coords['x']
    furthest_left = current_idx
    furthest_right = current_idx
    
    # Find the furthest left grid in the segment
    while furthest_left > 0 and row_grids[furthest_left - 1]['coords']['x'] == x - grid_size:
        furthest_left -= 1
        x -= grid_size
    
    # Find the furthest right grid in the segment
    x = current_coords['x']
    while furthest_right < len(row_grids) - 1 and row_grids[furthest_right + 1]['coords']['x'] == x + grid_size:
        furthest_right += 1
        x += grid_size
    
    segment_width = furthest_right - furthest_left
    if segment_width == 0:
        return 0
        
    position_ratio = (current_idx - furthest_left) / segment_width
    return 2 * abs(position_ratio - 0.5)
    

def get_penalty_colour(penalty: float) -> tuple[int, int, int]:
    """
    Get the colour based on the penalty value.    
    """
    closest_key = min(penalty_colour_gradient.keys(), key=lambda x: abs(x - penalty))
    return penalty_colour_gradient[closest_key]