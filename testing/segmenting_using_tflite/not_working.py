import cv2
import math
import numpy as np
import torch
import tensorflow as tf
import time

import testing.segmenting_using_tflite.ops as ops


def postprocess(preds, img, orig_imgs):
        """Applies non-max suppression and processes detections for each image in an input batch."""
        p = ops.non_max_suppression(
            preds[0],
            conf_thres=0.5,
            iou_thres=0.5,
            agnostic=False,
            max_det=300,
            nc=1,
            classes=[0],
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]  # tuple if PyTorch model or array if exported
        for i, (pred, orig_img) in enumerate(zip(p, orig_imgs)):
            if not len(pred):  # save empty boxes
                masks = None
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append((orig_img, ["sidewalk"], pred[:, :6], masks))
        return results


video = "../videos/bin_obstacles.MP4"
weights = "../runs/train11/weights/best_saved_model/best_float32.tflite"

interpreter = tf.lite.Interpreter(model_path=weights)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(video)
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
fps, fourcc = cap.get(cv2.CAP_PROP_FPS), cv2.VideoWriter_fourcc(*'mp4v')

ret, frame = cap.read()

original_frame = frame.copy()

frame = cv2.resize(frame, (640, 640))
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame = tf.convert_to_tensor(frame, dtype=tf.float32)
frame = tf.expand_dims(frame, 0)

# cv2.imshow("frame", frame)
# cv2.waitKey(0)

model_height = input_details[0]['shape'][1]
model_width = input_details[0]['shape'][2]

original_height, original_width = frame.shape[:2]

confidence_thres = 0.5
iou_thres = 0.5

interpreter.set_tensor(input_details[0]["index"], frame)
interpreter.invoke()

preds, protos = interpreter.get_tensor(output_details[0]["index"]), interpreter.get_tensor(output_details[1]["index"])

results = postprocess((preds, protos), frame, original_frame)
