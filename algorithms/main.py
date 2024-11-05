import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO

from FrameProcessor import FrameProcessor


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n-seg.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='../videos/longer.MP4', help='video file path')
    parser.add_argument('--output', type=str, default="../results/", help='output directory')
    # parser.add_argument('--save-frames', type=bool, default=False, help='save the frames from the output video')
    
    return parser.parse_args()


def main(weights: str | Path = 'yolov8n-seg.pt',
         source: str | Path = '../videos/longer.MP4',
         output: str | Path = '../results/',
         #save_frames: bool = False
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
    processor = FrameProcessor()
   
    # Setup video capture
    cap = cv2.VideoCapture(source)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
   
    # Setup output directory and path
    save_dir = Path(output)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{Path(source).stem}_experimental.mp4"
   
    # Delete video if it already exists
    if save_path.exists():
        save_path.unlink()
       
    # if save_frames:
    #     frames_dir = save_dir / f"{Path(source).stem}_experimental_frames"
    #     frames_dir.mkdir(parents=True, exist_ok=True)
       
    # Initialize video writer
    video_writer = cv2.VideoWriter(
        str(save_path), 
        fourcc, 
        fps, 
        (frame_width, frame_height)
    )
   
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
               
            start_time = cv2.getTickCount()
           
            # Use the frame processor instead of process_frame
            processed_frame = processor(frame, model)
           
            # cv2.imshow("Processed Frame", processed_frame)
            # cv2.waitKey(1)
           
            # if save_frames:
            #     frame_path = frames_dir / f"{cap.get(1)}.png"
            #     cv2.imwrite(str(frame_path), processed_frame)
           
            end_time = cv2.getTickCount()
           
            processing_time = (end_time - start_time) / cv2.getTickFrequency()
            print(f"Processing time: {processing_time} seconds")
           
            video_writer.write(processed_frame)
    
    finally:
        # Ensure resources are properly released
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
   

if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))