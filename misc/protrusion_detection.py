def detect_protrusions(frame: np.ndarray) -> List:
    """
    Detect protrusions in a binary or grayscale image.
    Uses stricter criteria to only detect significant protrusions.
    """
    frame_height, frame_width = frame.shape[:2]
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # Threshold if not already binary
    _, binary = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
   
    # Find global corner peak - highest point in the image that is on the path
    global_peak = find_corner_peak(binary)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    contour = max(contours, key=cv2.contourArea)
    c_x, c_y, c_w, c_h = cv2.boundingRect(contour)
    
    triangle = np.array([
        [c_x, c_y + c_h],
        [c_x + c_w, c_y + c_h],
        global_peak
    ])
    
    hull = cv2.convexHull(contour, returnPoints=True)
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull_indices)
    
    protrusions = []
    processed_regions = []
    
    for defect in defects:
        start, end, far, depth = defect[0]
        start = tuple(contour[start][0])
        end = tuple(contour[end][0])
        far = tuple(contour[far][0])
        
        v1 = np.array(start) - np.array(far)
        v2 = np.array(end) - np.array(far)
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle_deg = np.degrees(angle)
        
        # Stricter criteria for protrusion detection:
        # 1. Increased width threshold to 35% of contour width
        # 2. Narrowed angle range to detect sharper protrusions
        # 3. Added height-based filtering
        is_significant = (
            depth > 0.35 * c_w and  
            angle_deg > 45 and      
            angle_deg < 120 and     
            start[1] < c_y + (0.7 * c_h)
        )
        
        if not is_significant:
            continue
        
        # Increased minimum distance between protrusions
        is_new_region = True
        min_distance = max(c_w // 4, 150)  # At least 25% of contour width or 150 pixels
        
        for prev_start, prev_end in processed_regions:
            if (abs(start[0] - prev_start[0]) < min_distance and 
                abs(start[1] - prev_start[1]) < min_distance):
                is_new_region = False
                break
            
        if is_new_region:
            # Adjusted box size for peak detection
            box_height = frame_height // 8  # More focused vertical search
            box_width = frame_width // 4    # Wider horizontal search
            region_mask = get_protrusion_region(binary, start, box_size=(box_height, box_width))
            peak = find_corner_peak(binary, region_mask)
            
            if peak is not None and peak != global_peak:
                # Additional height check for the peak
                if peak[1] < c_y + (0.6 * c_h):  # Peak must be in top 60% of contour
                    protrusions.append(peak)
                    processed_regions.append((start, end))
    
    # # Debug visualization
    # debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 2)
    # cv2.drawContours(debug_img, [hull], -1, (255, 0, 0), 2)
    
    # # Draw detected peaks
    # if global_peak:
    #     cv2.circle(debug_img, global_peak, 5, (0, 0, 255), -1)
    # for peak in protrusions:
    #     cv2.circle(debug_img, peak, 5, (255, 255, 0), -1)
        
    # cv2.imshow("Protrusion Detection", debug_img)
    # cv2.waitKey(1)
    
    return [global_peak] + protrusions