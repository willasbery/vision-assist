def detect_protrusions(frame: np.ndarray, all_grids: List) -> List:
    """
    Detect protrusions in a binary or grayscale image.
    Uses stricter criteria to only detect significant protrusions.
    Filters out protrusions near quadrilateral edges.
    """
    frame_height, frame_width = frame.shape[:2]
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    _, binary = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
   
    # Find global corner peak - highest point in the image that is on the path
    global_peak, global_peak_min_x, global_peak_max_x = find_corner_peak(binary, return_min_max=True)
   
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    if len(contours) == 0:
        return None
   
    contour = max(contours, key=cv2.contourArea)
    c_x, c_y, c_w, c_h = cv2.boundingRect(contour)
    
    hull = cv2.convexHull(contour, returnPoints=True)
    
    # OPTIMISATION: use the quadtrilateral formed by the hull to remove any paths falsely
    # detected as protrusions
    #   ---------------
    #  |               |
    #  |    C    D     |
    #  |    --*---     |
    #  |   /      \    |
    #  |  |        \   |
    #  |  /         \* | <-- Falsely detected protrusion
    #  | /           \ |
    # B|/             \|A
    #   ---------------
    # In the example above, we would remove the falsely detected protrusion
    # as it is within the threshold of the hull formed by the path (ABCD)
    hull_points = np.array([point[0] for point in hull])
    
    # TODO: this can be further optimised, we ideally want the bottom left and bottom right points,
    # and these MUST have an entire column of cells below them until the frame_height, otherwise they are not valid
    
    def check_full_column_below(point, all_grids):
        # grid that the point is nearest to
        closest_grid = sorted(all_grids, key=lambda grid: np.linalg.norm(np.array(grid) - np.array(point)))[0]
        i, j = closest_grid
        
        # if any possible grids exceed the frame height, return True
        if j + grid_size >= frame_height:
            return True
        
        # get all grids in the same column and below the point in the y-column
        grids_in_column = [grid for grid in all_grids if grid[0] == i and grid[1] > j]
        # sort the grids by y-coordinate to slightly optimise the search
        grids_in_column.sort(key=lambda grid: grid[1])
        
        # simply iterate through each possible grid below the point UP TO 90% OF THE FRAME HEIGHT and check if it is in the grids
        while i < frame_height * 0.9:
            if (i, j) not in grids_in_column:
                return False
            i += grid_size
            
        return True
        
    # sort the hull points by the smallest x and smallest y coordinates
    sorted_for_bottom_left = hull_points[np.lexsort((hull_points[:, 1], hull_points[:, 0]))]
    # remove any points that are not on the left hand side of the frame#
    bottom_left_candidates = [point for point in sorted_for_bottom_left if point[0] < frame_width // 2]
    
    bottom_left = None
    for point in bottom_left_candidates:
        if check_full_column_below(point, all_grids):
            bottom_left = point
            break
        
    if bottom_left is None:
        bottom_left = bottom_left_candidates[0]
    
    # sort the hull points by smallest y coordinate and largest x coordinate
    sorted_for_bottom_right = hull_points[np.lexsort((hull_points[:, 1], -hull_points[:, 0]))]
    # remove any points that are not on the right hand side of the frame
    bottom_right_candidates = [point for point in sorted_for_bottom_right if point[0] > frame_width // 2]
    
    bottom_right = None
    for point in bottom_right_candidates:
        if check_full_column_below(point, all_grids):
            bottom_right = point
            break
        
    if bottom_right is None:
        bottom_right = bottom_right_candidates[0]
    
    quadrilateral_encompassing_path = np.array([
        bottom_left,
        bottom_right,
        global_peak_max_x,
        global_peak_min_x
    ])
   
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull_indices)
   
    protrusions = []
    processed_regions = []
    
    # for debug
    debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    cv2.polylines(debug_img, [quadrilateral_encompassing_path], True, (0, 255, 255), 2)
   
    for defect in defects:
        start, end, far, depth = defect[0]
        start = tuple(contour[start][0])
        end = tuple(contour[end][0])
        far = tuple(contour[far][0])
       
        v1 = np.array(start) - np.array(far)
        v2 = np.array(end) - np.array(far)
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle_deg = np.degrees(angle)
       
        is_significant = (
            depth > 0.35 * c_w and  
            angle_deg > 45 and      
            angle_deg < 120 and    
            start[1] < c_y + (0.7 * c_h)
        )
       
        if not is_significant:
            continue
       
        is_new_region = True
        min_distance = max(c_w // 4, 150)       
        for prev_start, prev_end in processed_regions:
            if (abs(start[0] - prev_start[0]) < min_distance and
                abs(start[1] - prev_start[1]) < min_distance):
                is_new_region = False
                break
           
        if is_new_region:
            box_height = frame_height // 8
            box_width = frame_width // 4
            region_mask = get_protrusion_region(binary, start, box_size=(box_height, box_width))
            peak = find_corner_peak(binary, region_mask)
           
            if peak is not None and peak != global_peak:
                if not is_point_near_quadrilateral(peak, quadrilateral_encompassing_path, threshold=50):
                    protrusions.append(peak)
                    processed_regions.append((start, end))
                    # Draw accepted peak in green
                    cv2.circle(debug_img, peak, 5, (0, 255, 0), -1)
                else:
                    # Draw rejected peak in red
                    cv2.circle(debug_img, peak, 5, (0, 0, 255), -1)
    
    # Draw global peak
    if global_peak:
        cv2.circle(debug_img, global_peak, 7, (255, 255, 0), -1)
    
    cv2.imshow("Protrusion Detection", debug_img)
    cv2.waitKey(1)
    
    return [global_peak] + protrusions