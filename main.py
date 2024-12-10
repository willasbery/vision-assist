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
    # parser.add_argument('--save-frames', type=bool, default=False, help='save the frames from the output video')
    parser.add_argument('--verbose', type=bool, default=False, help='print debug information')
    
    return parser.parse_args()


def main(weights: str | Path = 'yolov8n-seg.pt',
         source: str | Path = '../videos/longer.MP4',
         output: str | Path = '../results/',
         process_fps: int = 8,
         #save_frames: bool = False
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
   
    # # Setup output directory and path
    # save_dir = Path(output)
    # save_dir.mkdir(parents=True, exist_ok=True)
    # save_path = save_dir / f"{Path(source).stem}_experimental.mp4"
   
    # # Delete video if it already exists
    # if save_path.exists():
    #     save_path.unlink()
       
    # if save_frames:
    #     frames_dir = save_dir / f"{Path(source).stem}_experimental_frames"
    #     frames_dir.mkdir(parents=True, exist_ok=True)
       
    # Initialize video writer
    # video_writer = cv2.VideoWriter(
    #     str(save_path), 
    #     fourcc, 
    #     fps, 
    #     (frame_width, frame_height)
    # )
    
    process_delay = 1.0 / process_fps
    last_process_time = 0
    
    frames_processed = 0
    frames_skipped = 0
    
    processing_times = []
   
    try:
        while mock_cam.isOpened():
            current_time = time.time()
            
            if current_time - last_process_time < process_delay:
                continue
            
            ret, frame = mock_cam.read()
            if not ret:
                break
               
            start_time = cv2.getTickCount()
            
            processed_frame = None
            while processed_frame is None:
                processed_frame = processor(frame)
                resized_processed_frame = cv2.resize(processed_frame, (576, 1024))
                
                cv2.imshow("Processed Frame", resized_processed_frame)
                cv2.waitKey(1)
                
                if isinstance(processed_frame, bool) and not processed_frame:
                    frames_skipped += 1
                    
                    if verbose:
                        print(f"Frame {frames_processed} was skipped as it was too blurry, trying next frame")
                        
                    ret, frame = mock_cam.read()
                    if not ret:
                        break
           
            
            frames_processed += 1
            last_process_time = current_time
            
            end_time = cv2.getTickCount()
           
            processing_time = (end_time - start_time) / cv2.getTickFrequency()
            processing_times.append(processing_time)
            print(f"Processing time: {processing_time} seconds")
            
            # cv2.imshow("Processed Frame", processed_frame)
            # cv2.waitKey(1)
           
            # if save_frames:
            #     frame_path = frames_dir / f"{cap.get(1)}.png"
            #     cv2.imwrite(str(frame_path), processed_frame)
           
            # video_writer.write(processed_frame)
    except KeyboardInterrupt:
        avg_processing_time = sum(processing_times) / len(processing_times)
        print("\nProcessing summary:")
        print(f"Average processing time: {avg_processing_time} seconds")
        print(f"Frames processed: {frames_processed}")
        print(f"Frames skipped: {frames_skipped}")
        
        mock_cam.release()
        cv2.destroyAllWindows()
    finally:
        # Ensure resources are properly released
        # cap.release()
        # video_writer.release()
        mock_cam.release()
        cv2.destroyAllWindows()
   

if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))