import argparse
import cv2
import numpy as np
import random
import shutil
import torch

from pathlib import Path
from ultralytics import YOLO

def main(weights='yolov8n-seg.pt', device='cpu', source='../../videos/longer.MP4', output='./results/'):
    # Load the segmentation model
    model = YOLO(f"{weights}")
    # model.to("cuda") if device == "cuda" else model.to("cpu")
    
    # Class names in model
    names = model.model.names
    
    cap = cv2.VideoCapture(source)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps, fourcc = int(cap.get(5)), cv2.VideoWriter_fourcc(*"mp4v")
    
    save_dir = Path(output)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{Path(source).stem}.mp4"
    
    if save_path.exists():
        Path.unlink(save_path)
    
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))
 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.predict(frame, conf=0.5)
        for result in results:
            if result.masks is None:
                continue
            
            for mask, box in zip(result.masks.xy, result.boxes):
                points = np.int32([mask])
                cv2.fillPoly(frame, points, [255, 0, 0])

        video_writer.write(frame)
           
    cap.release()
    video_writer.release()
    print(f"Video saved at {save_dir}")
    cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n-seg.pt', help='path to weights file')
    parser.add_argument('--device', type=str, default='cpu', help='device')
    parser.add_argument('--source', type=str, default='videos/longer.MP4', help='video file path')
    parser.add_argument('--output', type=str, default="./results/", help='output directory')
    
    return parser.parse_args()
    
if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))