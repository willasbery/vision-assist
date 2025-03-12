from pathlib import Path
from ultralytics import YOLO


weights_path = Path("./runs/train11/weights/best.pt")

model = YOLO(weights_path)
# model.to("cuda")

model.export(format="tflite")