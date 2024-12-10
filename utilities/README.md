# Grid Drawing Utility

Note: README mainly written by Claude.

A Python-based tool for creating and editing grid-based drawings, designed to help test and generate input data for path detection algorithms.

## Overview

This utility allows users to create binary grid-based drawings where each cell can be either filled or empty. The drawings are saved in two formats:

1. A PNG image file without grid lines
2. A NumPy array file containing the binary grid data

The utility features a simple GUI interface with drawing tools and file management capabilities.

## Features

- Interactive grid-based drawing interface
- Adjustable brush sizes (1x1 to 5x5)
- Load and edit existing drawings
- Save drawings in both image and grid data formats
- File selection menu for managing multiple drawings
- Visual brush size preview
- Real-time grid visualization

## Usage

### Starting the main application to test on your newly drawn Grids (see below on how to use that file)

For some reason that I can't be bothered to fix, go back to the main folder (seems to be an issue with Python modules):

```bash
$ vision-assist>
```

Run this command now:

```bash
$ vision-assist> python ./utilities/testing_main.py <name_of_testing_case> --input-dir ./utilities/examples
```

Here is an example from the test cases in the examples folder:

```bash
$ vision-assist> python ./utilities/testing_main.py insane_case --input-dir ./utilities/examples
```

### Starting the Grid drawing file

Ensure you are in the utilities folder as it saves them in the examples Path relative to your CWD.

```bash
cd ./utilities/
```

Run the utility using:

```bash
python generate_testing_grids.py
```

### Controls

- **Left Mouse Button**: Draw/Erase (click and drag)
- **Number Keys (1-5)**: Change brush size
- **S Key**: Save current drawing
- **ESC**: Create new file/Exit menus

### Drawing

1. Select "New File" or an existing file from the menu
2. Left-click to draw (fills cells)
3. Left-click on filled cells to erase
4. Use number keys to change brush size
5. Press 'S' to save your work

### File Management

- Files are saved in the `./examples` directory
- Each drawing creates two files:
  - `{filename}_img.png`: The visual representation
  - `{filename}_grids.npy`: The grid data for processing

### Image/Grid Specifications

- Drawing canvas: 360x640 pixels (DRAW_WIDTH x DRAW_HEIGHT)
- Saved image: 720x1280 pixels (SAVE_WIDTH x SAVE_HEIGHT)
- Grid size: 10 pixels for drawing (DRAW_GRID_SIZE)
- Grid size: 20 pixels for saved files (SAVE_GRID_SIZE)

## Output Files

The utility generates two files for each drawing:

1. `{filename}_img.png`: A clean image without grid lines, suitable for visual processing
2. `{filename}_grids.npy`: A NumPy array containing boolean values (True for filled cells, False for empty cells)
