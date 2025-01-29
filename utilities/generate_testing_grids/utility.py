import pygame
import numpy as np
import cv2
import time
import os
from pathlib import Path

# Set up constants
DRAW_WIDTH, DRAW_HEIGHT = 360, 640
SAVE_WIDTH, SAVE_HEIGHT = 720, 1280
DRAW_GRID_SIZE = 10
SAVE_GRID_SIZE = 20
MIN_HOLD_TIME_MS = 10
MAX_BRUSH_SIZE = 5
EXAMPLES_DIR = "./examples"

def create_grid_surface(width, height, grid_size):
    surface = pygame.Surface((width, height))
    surface.fill((0, 0, 0))
    for x in range(0, width, grid_size):
        for y in range(0, height, grid_size):
            pygame.draw.rect(surface, (100, 100, 100), (x, y, grid_size, grid_size), 1)
    return surface

def get_grid_position(mouse_x, mouse_y, grid_size):
    return (mouse_x // grid_size) * grid_size, (mouse_y // grid_size) * grid_size

def get_affected_cells(grid_x, grid_y, brush_size, grid_shape, grid_size):
    cells = []
    offset = brush_size // 2
    
    grid_index_x = grid_x // grid_size
    grid_index_y = grid_y // grid_size
    
    for dy in range(-offset, offset + 1):
        for dx in range(-offset, offset + 1):
            new_x = grid_index_x + dx
            new_y = grid_index_y + dy
            if (0 <= new_x < grid_shape[1] and 0 <= new_y < grid_shape[0]):
                cells.append((new_x, new_y))
    
    return cells

def fill_cells(grid_surface, grid_filled, cells, fill_mode, grid_size):
    for grid_index_x, grid_index_y in cells:
        grid_x = grid_index_x * grid_size
        grid_y = grid_index_y * grid_size
        
        if grid_filled[grid_index_y, grid_index_x] != fill_mode:
            if fill_mode:
                pygame.draw.rect(grid_surface, (255, 255, 255), 
                               (grid_x, grid_y, grid_size, grid_size))
            else:
                pygame.draw.rect(grid_surface, (0, 0, 0), 
                               (grid_x, grid_y, grid_size, grid_size))
                pygame.draw.rect(grid_surface, (100, 100, 100), 
                               (grid_x, grid_y, grid_size, grid_size), 1)
            
            grid_filled[grid_index_y, grid_index_x] = fill_mode

def scale_grid_for_saving(grid_filled):
    # Create a larger grid for saving
    save_grid = np.zeros((SAVE_HEIGHT // SAVE_GRID_SIZE, SAVE_WIDTH // SAVE_GRID_SIZE), dtype=bool)
    
    # Scale factor between draw and save grids
    scale_x = save_grid.shape[1] / grid_filled.shape[1]
    scale_y = save_grid.shape[0] / grid_filled.shape[0]
    
    # Map each cell from the drawing grid to the save grid
    for y in range(grid_filled.shape[0]):
        for x in range(grid_filled.shape[1]):
            if grid_filled[y, x]:
                save_x = int(x * scale_x)
                save_y = int(y * scale_y)
                save_grid[save_y, save_x] = True
    
    return save_grid

def save_image_without_grid(grid_filled, filename_base):
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    
    # Scale up the grid before saving
    save_grid = scale_grid_for_saving(grid_filled)
    
    surface_no_grid = pygame.Surface((SAVE_WIDTH, SAVE_HEIGHT))
    surface_no_grid.fill((0, 0, 0))

    for y in range(save_grid.shape[0]):
        for x in range(save_grid.shape[1]):
            if save_grid[y, x]:
                pygame.draw.rect(surface_no_grid, (255, 255, 255), 
                               (x * SAVE_GRID_SIZE, y * SAVE_GRID_SIZE, SAVE_GRID_SIZE, SAVE_GRID_SIZE))

    np_frame = pygame.surfarray.array3d(surface_no_grid).swapaxes(0, 1)
    image_filename = os.path.join(EXAMPLES_DIR, f"{filename_base}_img.png")
    cv2.imwrite(image_filename, np_frame)
    print(f"Image saved as {image_filename}")

def save_grid_data(grid_filled, filename_base):
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    # Save the scaled up version
    save_grid = scale_grid_for_saving(grid_filled)
    grid_filename = os.path.join(EXAMPLES_DIR, f"{filename_base}_grids.npy")
    np.save(grid_filename, save_grid)
    print(f"Grid data saved as {grid_filename}")

def load_grid_data(filename_base):
    grid_filename = os.path.join(EXAMPLES_DIR, f"{filename_base}_grids.npy")
    try:
        saved_grid = np.load(grid_filename)
        # Scale down for drawing
        draw_grid = np.zeros((DRAW_HEIGHT // DRAW_GRID_SIZE, DRAW_WIDTH // DRAW_GRID_SIZE), dtype=bool)
        
        scale_x = draw_grid.shape[1] / saved_grid.shape[1]
        scale_y = draw_grid.shape[0] / saved_grid.shape[0]
        
        for y in range(saved_grid.shape[0]):
            for x in range(saved_grid.shape[1]):
                if saved_grid[y, x]:
                    draw_x = int(x * scale_x)
                    draw_y = int(y * scale_y)
                    if draw_x < draw_grid.shape[1] and draw_y < draw_grid.shape[0]:
                        draw_grid[draw_y, draw_x] = True
        
        return draw_grid
    except FileNotFoundError:
        print(f"Could not find file: {grid_filename}")
        return None

def draw_text_centered(surface, text, font, color, y_offset=0):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=(DRAW_WIDTH/2, DRAW_HEIGHT/2 + y_offset))
    surface.blit(text_surface, text_rect)

def get_saved_files():
    if not os.path.exists(EXAMPLES_DIR):
        return []
    
    files = []
    for filename in os.listdir(EXAMPLES_DIR):
        if filename.endswith("_grids.npy"):
            files.append(filename[:-10])  # Remove '_grids.npy' from filename
    return sorted(files)

def show_file_selection_menu():
    pygame.font.init()
    font = pygame.font.Font(None, 36)
    font_large = pygame.font.Font(None, 48)
    screen = pygame.display.get_surface()
    saved_files = get_saved_files()
    
    menu_options = ["[New File]"] + saved_files
    selected_option = 0
    scroll_offset = 0
    max_visible_items = 10
    
    while True:
        screen.fill((0, 0, 0))
        
        # Draw title
        draw_text_centered(screen, "Grid Drawing Utility", font_large, (255, 255, 255), -250)
        draw_text_centered(screen, "Select a file to load or create a new one", font, (200, 200, 200), -180)
        
        # Draw instructions
        instructions = [
            "↑/↓: Navigate",
            "Enter: Select",
            "Esc: New File"
        ]
        for i, instruction in enumerate(instructions):
            draw_text_centered(screen, instruction, font, (150, 150, 150), 200 + i * 40)
        
        # Draw file options
        visible_options = menu_options[scroll_offset:scroll_offset + max_visible_items]
        for i, option in enumerate(visible_options):
            color = (255, 255, 0) if i + scroll_offset == selected_option else (255, 255, 255)
            draw_text_centered(screen, option, font, color, -100 + i * 40)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                elif event.key == pygame.K_RETURN:
                    if selected_option == 0:
                        return None
                    else:
                        return menu_options[selected_option]
                elif event.key == pygame.K_UP:
                    selected_option = (selected_option - 1) % len(menu_options)
                    if selected_option < scroll_offset:
                        scroll_offset = selected_option
                elif event.key == pygame.K_DOWN:
                    selected_option = (selected_option + 1) % len(menu_options)
                    if selected_option >= scroll_offset + max_visible_items:
                        scroll_offset = selected_option - max_visible_items + 1

def create_initial_grid_surface(grid_filled):
    surface = create_grid_surface()
    for y in range(grid_filled.shape[0]):
        for x in range(grid_filled.shape[1]):
            if grid_filled[y, x]:
                grid_x = x * DRAW_GRID_SIZE
                grid_y = y * DRAW_GRID_SIZE
                pygame.draw.rect(surface, (255, 255, 255), 
                               (grid_x, grid_y, DRAW_GRID_SIZE, DRAW_GRID_SIZE))
    return surface

def get_filename_input():
    pygame.font.init()
    font = pygame.font.Font(None, 36)
    
    input_surface = pygame.Surface((400, 50))
    current_text = ""
    
    while True:
        input_surface.fill((50, 50, 50))
        text_surface = font.render(current_text + "|", True, (255, 255, 255))
        input_surface.blit(text_surface, (10, 10))
        
        pygame.display.get_surface().fill((0, 0, 0))
        pygame.display.get_surface().blit(input_surface, (DRAW_WIDTH//2 - 200, DRAW_HEIGHT//2 - 25))
        
        prompt_text = font.render("Enter filename base (press Enter when done):", True, (255, 255, 255))
        pygame.display.get_surface().blit(prompt_text, (DRAW_WIDTH//2 - 200, DRAW_HEIGHT//2 - 60))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    return current_text if current_text else "default"
                elif event.key == pygame.K_BACKSPACE:
                    current_text = current_text[:-1]
                else:
                    if event.unicode.isalnum() or event.unicode in "-_":
                        current_text += event.unicode

def create_initial_grid_surface(grid_filled):
    surface = create_grid_surface(DRAW_WIDTH, DRAW_HEIGHT, DRAW_GRID_SIZE)
    for y in range(grid_filled.shape[0]):
        for x in range(grid_filled.shape[1]):
            if grid_filled[y, x]:
                grid_x = x * DRAW_GRID_SIZE
                grid_y = y * DRAW_GRID_SIZE
                pygame.draw.rect(surface, (255, 255, 255), 
                               (grid_x, grid_y, DRAW_GRID_SIZE, DRAW_GRID_SIZE))
    return surface

def draw_brush_preview(screen, brush_size, mouse_pos):
    preview_surface = pygame.Surface((DRAW_WIDTH, DRAW_HEIGHT), pygame.SRCALPHA)
    grid_x, grid_y = get_grid_position(*mouse_pos, DRAW_GRID_SIZE)
    offset = (brush_size // 2) * DRAW_GRID_SIZE
    
    preview_rect = pygame.Rect(
        grid_x - offset,
        grid_y - offset,
        DRAW_GRID_SIZE * brush_size,
        DRAW_GRID_SIZE * brush_size
    )
    pygame.draw.rect(preview_surface, (255, 255, 255, 64), preview_rect)
    screen.blit(preview_surface, (0, 0))

def main():
    pygame.init()
    screen = pygame.display.set_mode((DRAW_WIDTH, DRAW_HEIGHT))
    pygame.display.set_caption("Grid Drawing Utility")
    clock = pygame.time.Clock()
    
    # Show file selection menu
    selected_file = show_file_selection_menu()
    
    # Load existing file or create new grid
    if selected_file:
        grid_filled = load_grid_data(selected_file)
        if grid_filled is None:
            grid_filled = np.zeros((DRAW_HEIGHT // DRAW_GRID_SIZE, DRAW_WIDTH // DRAW_GRID_SIZE), dtype=bool)
        grid_surface = create_initial_grid_surface(grid_filled)
    else:
        grid_filled = np.zeros((DRAW_HEIGHT // DRAW_GRID_SIZE, DRAW_WIDTH // DRAW_GRID_SIZE), dtype=bool)
        grid_surface = create_grid_surface(DRAW_WIDTH, DRAW_HEIGHT, DRAW_GRID_SIZE)
    
    drawing = False
    last_mouse_down_time = 0
    fill_mode = None
    brush_size = 1
    font = pygame.font.Font(None, 36)

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    drawing = True
                    last_mouse_down_time = time.time() * 1000
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    grid_x, grid_y = get_grid_position(mouse_x, mouse_y, DRAW_GRID_SIZE)
                    
                    grid_index_x = grid_x // DRAW_GRID_SIZE
                    grid_index_y = grid_y // DRAW_GRID_SIZE
                    
                    if 0 <= grid_index_x < grid_filled.shape[1] and 0 <= grid_index_y < grid_filled.shape[0]:
                        fill_mode = not grid_filled[grid_index_y, grid_index_x]
                        cells = get_affected_cells(grid_x, grid_y, brush_size, grid_filled.shape, DRAW_GRID_SIZE)
                        fill_cells(grid_surface, grid_filled, cells, fill_mode, DRAW_GRID_SIZE)
                        
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    current_time = time.time() * 1000
                    if current_time - last_mouse_down_time >= MIN_HOLD_TIME_MS:
                        drawing = False
                    fill_mode = None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    filename_base = get_filename_input()
                    if filename_base:
                        save_image_without_grid(grid_filled, filename_base)
                        save_grid_data(grid_filled, filename_base)
                elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5]:
                    brush_size = int(event.unicode)

        if drawing:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            grid_x, grid_y = get_grid_position(mouse_x, mouse_y, DRAW_GRID_SIZE)
            cells = get_affected_cells(grid_x, grid_y, brush_size, grid_filled.shape, DRAW_GRID_SIZE)
            fill_cells(grid_surface, grid_filled, cells, fill_mode, DRAW_GRID_SIZE)

        screen.blit(grid_surface, (0, 0))
        draw_brush_preview(screen, brush_size, mouse_pos)
        
        brush_text = font.render(f"Brush Size: {brush_size}x{brush_size}", True, (255, 255, 255))
        screen.blit(brush_text, (10, 10))
        
        if selected_file:
            file_text = font.render(f"Current File: {selected_file}", True, (255, 255, 255))
            screen.blit(file_text, (10, 50))
        
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()