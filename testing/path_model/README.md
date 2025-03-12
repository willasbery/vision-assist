# Path Processing Testing

This folder contains tests and utilities for verifying the path processing functionality, particularly focused on path sectioning and corner detection.

## Files Overview

### Core Test Files
- `test.py`: Main test visualization script that processes sample paths and displays sections/corners
- `grids.py`: Contains sample grid data for testing various path configurations
- `other_models.py`: Base model definitions (Coordinate, Grid, Corner, Obstacle)
- `path_model.py`: Core Path model implementation with section and corner detection

## Functionality

### Grid Generation and Testing
- Uses predefined grid arrays that represent different path scenarios
- Tests path sectioning, distinguishing between straight and curved sections
- Visualizes corners and their properties (direction, sharpness, angle)

### Visualization Features
- Creates visual representations of paths with colored sections
- Draws corner markers with detailed information
- Saves processed images with path visualization
- Uses different colors to distinguish between section types:
  - Red (255, 0, 0)
  - Green (0, 255, 0)
  - Blue (0, 0, 255)

## Running Tests

To run the visualization tests:
```bash
python test.py
```

This will:
1. Load each predefined path from `grids.py`
2. Process the path into sections
3. Detect corners
4. Display the visualization window
5. Save images with format `{start_x}_{start_y}.png`

Press any key to proceed through each test visualization.

## Test Output

Each test generates:
- Console output for section and corner detection
- Visual window showing the processed path
- Saved image files for each test case

### Image Output Format
- Resolution: 576x1024 (scaled from 720x1280)
- Shows:
  - Path sections in different colors
  - Corner markers with text labels
  - Grid-based structure

## Notes

- Grid size is set to 20 pixels
- Sections with less than 4 grids are merged with adjacent sections
- Straight sections require 5 or more aligned grids
- Corner detection includes both sharp (>45°) and sweeping (≤45°) corners