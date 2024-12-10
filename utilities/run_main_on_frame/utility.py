import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import cv2
import time
from pathlib import Path
from ultralytics import YOLO

from FrameProcessor import FrameProcessor


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n-seg.pt', help='path to weights file')
    parser.add_argument('--source-frame', type=str, default='../frames/tester.jpg', help='video frame file path')
    return parser.parse_args()


def main(weights: str | Path = 'yolov8n-seg.pt',
         source_frame: str | Path = '../videos/longer.MP4',
         verbose=False
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
   
    frame = cv2.imread(source_frame)
    if frame is None:
        raise ValueError(f"Could not read frame file: {source_frame}")
        
    processing_times = []
   
    try:   
        start_time = cv2.getTickCount()
        
        processed_frame = None
        while processed_frame is None:
            processed_frame = processor(frame)
            resized_processed_frame = cv2.resize(processed_frame, (576, 1024))
            
            cv2.imshow("Processed Frame", resized_processed_frame)
            cv2.waitKey(0)
                    
        end_time = cv2.getTickCount()
        
        processing_time = (end_time - start_time) / cv2.getTickFrequency()
        processing_times.append(processing_time)
        print(f"Processing time: {processing_time} seconds")
    except KeyboardInterrupt:
        avg_processing_time = sum(processing_times) / len(processing_times)
        print("\nProcessing summary:")
        print(f"Average processing time: {avg_processing_time} seconds")
        
        cv2.destroyAllWindows()
    finally:
        # Ensure resources are properly released
        # cap.release()
        # video_writer.release()
        cv2.destroyAllWindows()
   

if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))