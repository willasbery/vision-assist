from ultralytics import YOLO
import torch

# Getting this to work on cuda was fun. Good luck if you're trying to run this.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    # Google said to do this lol.
    torch.multiprocessing.freeze_support()
    
    # build the model using gpu
    model = YOLO('yolo11n-seg.pt').to("cuda")  # Use the 'yolov8n-seg' model for segmentation
    
    results = model.train(data="data.yaml", epochs=150, imgsz=240) # train the model