import pygame
import numpy as np
import cv2
import time

# Set up constants
SCREEN_WIDTH, SCREEN_HEIGHT = 720, 1280
GRID_SIZE = 20
MIN_HOLD_TIME_MS = 10  # Minimum hold time to distinguish between click and hold

def create_grid_surface():
    surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    surface.fill((0, 0, 0))  # Fill background with black
    for x in range(0, SCREEN_WIDTH, GRID_SIZE):
        for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
            pygame.draw.rect(surface, (100, 100, 100), (x, y, GRID_SIZE, GRID_SIZE), 1)  # Grid lines
    return surface

def get_grid_position(mouse_x, mouse_y):
    return (mouse_x // GRID_SIZE) * GRID_SIZE, (mouse_y // GRID_SIZE) * GRID_SIZE

def save_image_without_grid(grid_filled):
    # Create a blank surface for saving without grid lines
    surface_no_grid = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    surface_no_grid.fill((0, 0, 0))  # Fill background with black

    # Fill only the highlighted grids
    for y in range(grid_filled.shape[0]):
        for x in range(grid_filled.shape[1]):
            if grid_filled[y, x]:  # If this grid cell is filled
                pygame.draw.rect(surface_no_grid, (255, 255, 255), 
                                 (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

    # Convert the Pygame surface to a numpy array and save it using OpenCV
    np_frame = pygame.surfarray.array3d(surface_no_grid).swapaxes(0, 1)
    cv2.imwrite("./examples/segmented_grid_no_gridlines.png", np_frame)
    print("Image saved as segmented_grid_no_gridlines.png")

def save_grid_data(grid_filled):
    # Save the grid data as a numpy array file
    np.save("./examples/grid_data.npy", grid_filled)
    print("Grid data saved as grid_data.npy")

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Grid Drawing Utility")
    clock = pygame.time.Clock()
    
    grid_surface = create_grid_surface()
    grid_filled = np.zeros((SCREEN_HEIGHT // GRID_SIZE, SCREEN_WIDTH // GRID_SIZE), dtype=bool)
    
    drawing = False
    last_mouse_down_time = 0
    fill_mode = None  # None means no fill mode set, True to fill, False to unfill

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    drawing = True
                    last_mouse_down_time = time.time() * 1000  # Convert to milliseconds
                    # Determine fill mode on initial click
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    grid_x, grid_y = get_grid_position(mouse_x, mouse_y)
                    grid_index_x = grid_x // GRID_SIZE
                    grid_index_y = grid_y // GRID_SIZE
                    
                    # Ensure the indices are within bounds
                    if 0 <= grid_index_x < grid_filled.shape[1] and 0 <= grid_index_y < grid_filled.shape[0]:
                        fill_mode = not grid_filled[grid_index_y, grid_index_x]

                        # Fill or unfill the clicked grid based on initial state
                        if fill_mode:
                            pygame.draw.rect(grid_surface, (255, 255, 255), (grid_x, grid_y, GRID_SIZE, GRID_SIZE))
                        else:
                            pygame.draw.rect(grid_surface, (0, 0, 0), (grid_x, grid_y, GRID_SIZE, GRID_SIZE))
                            pygame.draw.rect(grid_surface, (100, 100, 100), (grid_x, grid_y, GRID_SIZE, GRID_SIZE), 1)
                        
                        grid_filled[grid_index_y, grid_index_x] = fill_mode
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left click
                    current_time = time.time() * 1000  # Convert to milliseconds
                    # Only set drawing to False if more than MIN_HOLD_TIME_MS has passed
                    if current_time - last_mouse_down_time >= MIN_HOLD_TIME_MS:
                        drawing = False
                    fill_mode = None  # Reset fill mode after mouse is released
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                # Save the grid image without grid lines and grid data
                save_image_without_grid(grid_filled)
                save_grid_data(grid_filled)

        # Update grid filling when mouse is held down (for dragging effect)
        if drawing:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            grid_x, grid_y = get_grid_position(mouse_x, mouse_y)
            grid_index_x = grid_x // GRID_SIZE
            grid_index_y = grid_y // GRID_SIZE
            
            # Ensure the indices are within bounds before applying fill state
            if 0 <= grid_index_x < grid_filled.shape[1] and 0 <= grid_index_y < grid_filled.shape[0]:
                # Apply fill only if the cell's current state differs from fill_mode
                if grid_filled[grid_index_y, grid_index_x] != fill_mode:
                    if fill_mode:
                        pygame.draw.rect(grid_surface, (255, 255, 255), (grid_x, grid_y, GRID_SIZE, GRID_SIZE))
                    else:
                        pygame.draw.rect(grid_surface, (0, 0, 0), (grid_x, grid_y, GRID_SIZE, GRID_SIZE))
                        pygame.draw.rect(grid_surface, (100, 100, 100), (grid_x, grid_y, GRID_SIZE, GRID_SIZE), 1)
                    
                    grid_filled[grid_index_y, grid_index_x] = fill_mode

        # Draw the updated grid surface
        screen.blit(grid_surface, (0, 0))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
