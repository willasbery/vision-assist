

import argparse
import cv2
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
         #save_frames: bool = False
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
    # Mock camera for testing
    mock_cam = MockCamera(source, target_fps=30)
   
    try:
        while mock_cam.isOpened():
            ret, frame = mock_cam.read()
            if not ret:
                break
               
            start_time = cv2.getTickCount()
           
            # get the blur level of the frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_level = cv2.Laplacian(frame, cv2.CV_64F).var()
            
            if blur_level < 150:
                print("Blurry frame detected")
                cv2.putText(frame, "Blurry frame detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("Frame", frame)
                cv2.waitKey(0)
                
            # Display the frame with blur level
            
                
            cv2.putText(frame, f"Blur level: {blur_level}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
           
            end_time = cv2.getTickCount()
           
            processing_time = (end_time - start_time) / cv2.getTickFrequency()
            print(f"Processing time: {processing_time} seconds")
           
            # video_writer.write(processed_frame)
    
    finally:
        # Ensure resources are properly released
        # cap.release()
        # video_writer.release()
        mock_cam.release()
        cv2.destroyAllWindows()
   

if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))