import argparse
import cv2
import pygame
import os
from pathlib import Path as PathLibPath

def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Process grid data and visualize paths.')
    parser.add_argument('--video-path', type=str, default='./test.mp4',
                       help='File for frame extraction (default: ./test.mp4)')
    return parser.parse_args()

def create_button(surface, text, position, size=(100, 40), color=(100, 100, 100)):
    button = pygame.Surface(size)
    button.fill(color)
    
    # Create text
    font = pygame.font.Font(None, 32)
    text_surface = font.render(text, True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=(size[0]/2, size[1]/2))
    
    # Blit text onto button
    button.blit(text_surface, text_rect)
    
    # Get button rect for click detection
    button_rect = surface.blit(button, position)
    return button_rect

def scale_frame_to_fixed_size(frame, target_height=972, target_width=1728):  # 90% of 1080x1920
    # Calculate aspect ratios
    target_aspect = target_width / target_height
    frame_aspect = frame.shape[1] / frame.shape[0]
    
    # Determine dimensions to maintain aspect ratio
    if frame_aspect > target_aspect:
        # Width limited
        new_width = target_width
        new_height = int(target_width / frame_aspect)
    else:
        # Height limited
        new_height = target_height
        new_width = int(target_height * frame_aspect)
    
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

def skip_frames(cap, num_frames):
    """Skip specified number of frames and return the last frame"""
    for _ in range(num_frames - 1):
        cap.read()  # Read and discard frames
    return cap.read()  # Return the last frame

def process_video(video_path: str):
    # Initialize Pygame
    pygame.init()
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Create output directory based on video name
    video_name = PathLibPath(video_path).stem
    output_dir = PathLibPath(f"./extracted_frames_{video_name}")
    output_dir.mkdir(exist_ok=True)
    
    # Fixed display dimensions (90% of 1080x1920)
    DISPLAY_HEIGHT = 864  # 1080 * 0.8
    DISPLAY_WIDTH = 1200  
    
    # Set up display
    window_width = DISPLAY_WIDTH
    window_height = DISPLAY_HEIGHT + 100  # Extra space for buttons
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Frame Extractor")
    
    frame_count = 0
    running = True
    
    while running:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break
            
        # Scale frame to fixed size while preserving aspect ratio
        scaled_frame = scale_frame_to_fixed_size(frame, DISPLAY_HEIGHT, DISPLAY_WIDTH)
        
        # Convert frame from BGR to RGB
        frame_rgb = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)
        
        # Convert to pygame surface
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        
        # Clear screen
        screen.fill((200, 200, 200))
        
        # Calculate frame position to center it
        frame_x = (DISPLAY_WIDTH - frame_surface.get_width()) // 2
        frame_y = (DISPLAY_HEIGHT - frame_surface.get_height()) // 2
        
        # Display frame
        screen.blit(frame_surface, (frame_x, frame_y))
        
        # Create buttons - adjust positions for 4 buttons
        button_width = 120  # Slightly wider for text
        spacing = window_width // 5
        save_button = create_button(screen, "Save", (spacing - button_width//2, DISPLAY_HEIGHT + 30), size=(button_width, 40), color=(0, 150, 0))
        skip_button = create_button(screen, "Skip", (2*spacing - button_width//2, DISPLAY_HEIGHT + 30), size=(button_width, 40), color=(150, 0, 0))
        skip_10_button = create_button(screen, "Skip 10", (3*spacing - button_width//2, DISPLAY_HEIGHT + 30), size=(button_width, 40), color=(150, 50, 0))
        skip_100_button = create_button(screen, "Skip 100", (4*spacing - button_width//2, DISPLAY_HEIGHT + 30), size=(button_width, 40), color=(150, 100, 0))
        
        pygame.display.flip()
        
        # Handle events
        waiting_for_input = True
        while waiting_for_input:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    waiting_for_input = False
                    
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    if save_button.collidepoint(mouse_pos):
                        # Save original (unscaled) frame
                        output_path = output_dir / f"frame_{frame_count:04d}.png"
                        cv2.imwrite(str(output_path), frame)
                        print(f"Saved frame {frame_count} to {output_path}")
                        waiting_for_input = False
                        
                    elif skip_button.collidepoint(mouse_pos):
                        print(f"Skipped frame {frame_count}")
                        waiting_for_input = False
                        
                    elif skip_10_button.collidepoint(mouse_pos):
                        ret, frame = skip_frames(cap, 10)
                        if not ret:
                            print("End of video reached while skipping")
                            running = False
                            waiting_for_input = False
                        else:
                            frame_count += 9  # Add 9 because the loop will add 1
                            print(f"Skipped 10 frames to frame {frame_count + 1}")
                            waiting_for_input = False
                            
                    elif skip_100_button.collidepoint(mouse_pos):
                        ret, frame = skip_frames(cap, 100)
                        if not ret:
                            print("End of video reached while skipping")
                            running = False
                            waiting_for_input = False
                        else:
                            frame_count += 99  # Add 99 because the loop will add 1
                            print(f"Skipped 100 frames to frame {frame_count + 1}")
                            waiting_for_input = False
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    pygame.quit()

def main():
    args = setup_argparse()
    process_video(args.video_path)

if __name__ == "__main__":
    main()