import pygame
import numpy as np
import cv2
import time
import os

# Set up constants
SCREEN_WIDTH, SCREEN_HEIGHT = 720, 1280
GRID_SIZE = 20
MIN_HOLD_TIME_MS = 10
MAX_BRUSH_SIZE = 5
EXAMPLES_DIR = "./examples"

def create_grid_surface():
    surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    surface.fill((0, 0, 0))
    for x in range(0, SCREEN_WIDTH, GRID_SIZE):
        for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
            pygame.draw.rect(surface, (100, 100, 100), (x, y, GRID_SIZE, GRID_SIZE), 1)
    return surface

def get_grid_position(mouse_x, mouse_y):
    return (mouse_x // GRID_SIZE) * GRID_SIZE, (mouse_y // GRID_SIZE) * GRID_SIZE

def get_affected_cells(grid_x, grid_y, brush_size, grid_shape):
    cells = []
    offset = brush_size // 2
    
    grid_index_x = grid_x // GRID_SIZE
    grid_index_y = grid_y // GRID_SIZE
    
    for dy in range(-offset, offset + 1):
        for dx in range(-offset, offset + 1):
            new_x = grid_index_x + dx
            new_y = grid_index_y + dy
            if (0 <= new_x < grid_shape[1] and 0 <= new_y < grid_shape[0]):
                cells.append((new_x, new_y))
    
    return cells

def fill_cells(grid_surface, grid_filled, cells, fill_mode):
    for grid_index_x, grid_index_y in cells:
        grid_x = grid_index_x * GRID_SIZE
        grid_y = grid_index_y * GRID_SIZE
        
        if grid_filled[grid_index_y, grid_index_x] != fill_mode:
            if fill_mode:
                pygame.draw.rect(grid_surface, (255, 255, 255), 
                               (grid_x, grid_y, GRID_SIZE, GRID_SIZE))
            else:
                pygame.draw.rect(grid_surface, (0, 0, 0), 
                               (grid_x, grid_y, GRID_SIZE, GRID_SIZE))
                pygame.draw.rect(grid_surface, (100, 100, 100), 
                               (grid_x, grid_y, GRID_SIZE, GRID_SIZE), 1)
            
            grid_filled[grid_index_y, grid_index_x] = fill_mode

def save_image_without_grid(grid_filled, filename_base):
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    
    surface_no_grid = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    surface_no_grid.fill((0, 0, 0))

    for y in range(grid_filled.shape[0]):
        for x in range(grid_filled.shape[1]):
            if grid_filled[y, x]:
                pygame.draw.rect(surface_no_grid, (255, 255, 255), 
                               (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

    np_frame = pygame.surfarray.array3d(surface_no_grid).swapaxes(0, 1)
    image_filename = os.path.join(EXAMPLES_DIR, f"{filename_base}_img.png")
    cv2.imwrite(image_filename, np_frame)
    print(f"Image saved as {image_filename}")

def save_grid_data(grid_filled, filename_base):
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    grid_filename = os.path.join(EXAMPLES_DIR, f"{filename_base}_grids.npy")
    np.save(grid_filename, grid_filled)
    print(f"Grid data saved as {grid_filename}")

def load_grid_data(filename_base):
    grid_filename = os.path.join(EXAMPLES_DIR, f"{filename_base}_grids.npy")
    try:
        grid_filled = np.load(grid_filename)
        return grid_filled
    except FileNotFoundError:
        print(f"Could not find file: {grid_filename}")
        return None

def draw_text_centered(surface, text, font, color, y_offset=0):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2 + y_offset))
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
                grid_x = x * GRID_SIZE
                grid_y = y * GRID_SIZE
                pygame.draw.rect(surface, (255, 255, 255), 
                               (grid_x, grid_y, GRID_SIZE, GRID_SIZE))
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
        pygame.display.get_surface().blit(input_surface, (SCREEN_WIDTH//2 - 200, SCREEN_HEIGHT//2 - 25))
        
        prompt_text = font.render("Enter filename base (press Enter when done):", True, (255, 255, 255))
        pygame.display.get_surface().blit(prompt_text, (SCREEN_WIDTH//2 - 200, SCREEN_HEIGHT//2 - 60))
        
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

def draw_brush_preview(screen, brush_size, mouse_pos):
    preview_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    grid_x, grid_y = get_grid_position(*mouse_pos)
    offset = (brush_size // 2) * GRID_SIZE
    
    preview_rect = pygame.Rect(
        grid_x - offset,
        grid_y - offset,
        GRID_SIZE * brush_size,
        GRID_SIZE * brush_size
    )
    pygame.draw.rect(preview_surface, (255, 255, 255, 64), preview_rect)
    screen.blit(preview_surface, (0, 0))

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Grid Drawing Utility")
    clock = pygame.time.Clock()
    
    # Show file selection menu
    selected_file = show_file_selection_menu()
    
    # Load existing file or create new grid
    if selected_file:
        grid_filled = load_grid_data(selected_file)
        if grid_filled is None:
            grid_filled = np.zeros((SCREEN_HEIGHT // GRID_SIZE, SCREEN_WIDTH // GRID_SIZE), dtype=bool)
        grid_surface = create_initial_grid_surface(grid_filled)
    else:
        grid_filled = np.zeros((SCREEN_HEIGHT // GRID_SIZE, SCREEN_WIDTH // GRID_SIZE), dtype=bool)
        grid_surface = create_grid_surface()
    
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
                    grid_x, grid_y = get_grid_position(mouse_x, mouse_y)
                    
                    grid_index_x = grid_x // GRID_SIZE
                    grid_index_y = grid_y // GRID_SIZE
                    
                    if 0 <= grid_index_x < grid_filled.shape[1] and 0 <= grid_index_y < grid_filled.shape[0]:
                        fill_mode = not grid_filled[grid_index_y, grid_index_x]
                        cells = get_affected_cells(grid_x, grid_y, brush_size, grid_filled.shape)
                        fill_cells(grid_surface, grid_filled, cells, fill_mode)
                        
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
            grid_x, grid_y = get_grid_position(mouse_x, mouse_y)
            cells = get_affected_cells(grid_x, grid_y, brush_size, grid_filled.shape)
            fill_cells(grid_surface, grid_filled, cells, fill_mode)

        screen.blit(grid_surface, (0, 0))
        draw_brush_preview(screen, brush_size, mouse_pos)
        
        # Draw brush size indicator and current file name
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