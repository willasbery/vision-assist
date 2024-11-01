import heapq
from typing import Dict, List, Tuple


def a_star(graph: Dict[Tuple[int, int], List], start: Tuple[int, int], 
           goal: Tuple[int, int], penalties: Dict) -> Tuple[List, float]:
    """
    A* algorithm to find the shortest path between two nodes in a graph.
    
    Args:
        graph: Graph represented as an adjacency list
        start: Starting node
        goal: Goal node
        penalties: Dictionary of penalties for each grid (heuristic)
    
    Returns:
        Tuple containing the shortest path and the distance
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
                f_score = tentative_g_score + ((tentative_g_score * penalties[neighbour]) ** 3)
                f_scores[neighbour] = f_score
                previous_nodes[neighbour] = current_node
                heapq.heappush(queue, (f_score, neighbour))
    
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = previous_nodes[current]
    path.reverse()
    
    return path, g_scores[goal]


def calculate_row_penalty(x: int, min_x: int, max_x: int) -> float:
    """
    Calculate the penalty for a grid based on its position within a row.
    
    Args:
        x: x-coordinate of the grid
        min_x: x-coordinate of the leftmost grid in the row
        max_x: x-coordinate of the rightmost grid in the row
    
    Returns:
        Penalty value between 0 and 1
    """
    row_width = max_x - min_x
    if row_width == 0:
        return 0
    
    position_ratio = (x - min_x) / row_width
    penalty = 2 * abs(position_ratio - 0.5)
    return penalty