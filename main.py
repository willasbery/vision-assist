import argparse
import cv2
import time
from pathlib import Path
from ultralytics import YOLO

from FrameProcessor import FrameProcessor
from MockCamera import MockCamera


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n-seg.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='../videos/longer.MP4', help='video file path')
    parser.add_argument('--output', type=str, default="../results/", help='output directory')
    parser.add_argument('--process-fps', type=int, default=8, help='process frames per second')
    parser.add_argument('--verbose', type=bool, default=False, help='print debug information')
    
    return parser.parse_args()


def main(weights: str | Path = 'yolov8n-seg.pt',
         source: str | Path = '../videos/longer.MP4',
         output: str | Path = '../results/',
         process_fps: int = 8,
         verbose: bool = False
        ) -> None:
    """
    Main function to process the video and generate the output.
   
    Args:
        weights: Path to the weights file
        source: Path to the video file
        output: Output directory
    """
    # Initialize the model and frame processor
    model = YOLO(f"{weights}")
    processor = FrameProcessor(model=model, verbose=verbose)
   
    # Mock camera for testing
    mock_cam = MockCamera(source, target_fps=30)
    
    # Setup output directory for saved frames
    save_dir = Path(output)
    frames_dir = save_dir / f"{Path(source).stem}_saved_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    frames_processed = 0
    frames_skipped = 0
    frames_saved = 0
    save_counter = 0  # Counter for multiple saves of the same frame
    processing_times = []
   
    try:
        while mock_cam.isOpened():
            ret, frame = mock_cam.read()
            if not ret:
                break
               
            start_time = cv2.getTickCount()
            
            processed_frame = None
            while processed_frame is None:
                processed_frame = processor(frame)
                
                if isinstance(processed_frame, bool) and not processed_frame:
                    frames_skipped += 1
                    if verbose:
                        print(f"Frame {frames_processed} was skipped as it was too blurry, trying next frame")
                    ret, frame = mock_cam.read()
                    if not ret:
                        break
            
            if processed_frame is None:
                break
                
            resized_processed_frame = cv2.resize(processed_frame, (576, 1024))
            
            # Display loop - stay on current frame until Enter is pressed
            while True:
                cv2.imshow("Processed Frame", resized_processed_frame)
                key = cv2.waitKey(1) & 0xFF
                
                # 'S' key - save current frame
                if key == ord('s'):
                    frame_path = frames_dir / f"frame_{frames_processed:04d}_{save_counter:02d}.png"
                    cv2.imwrite(str(frame_path), processed_frame)
                    frames_saved += 1
                    save_counter += 1
                    if verbose:
                        print(f"Saved frame to {frame_path}")
                
                # Enter key - proceed to next frame
                elif key == 13:  # ASCII code for Enter
                    save_counter = 0  # Reset save counter for new frame
                    break
                
                # 'Q' key - quit program
                elif key == ord('q'):
                    raise KeyboardInterrupt
            
            frames_processed += 1
            
            end_time = cv2.getTickCount()
            processing_time = (end_time - start_time) / cv2.getTickFrequency()
            processing_times.append(processing_time)
            print(f"Processing time: {processing_time} seconds")
            
    except KeyboardInterrupt:
        if processing_times:
            avg_processing_time = sum(processing_times) / len(processing_times)
            print("\nProcessing summary:")
            print(f"Average processing time: {avg_processing_time} seconds")
            print(f"Frames processed: {frames_processed}")
            print(f"Frames skipped: {frames_skipped}")
            print(f"Frames saved: {frames_saved}")
        
        mock_cam.release()
        cv2.destroyAllWindows()
    finally:
        mock_cam.release()
        cv2.destroyAllWindows()
   

if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))