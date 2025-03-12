import argparse
import cv2
import numpy as np
import tensorflow as tf

from pathlib import Path

video = "../videos/bin_obstacles.MP4"
weights = "../runs/train11/weights/best_saved_model/best_float32.tflite"

interpreter = tf.lite.Interpreter(model_path=weights)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
# print(input_details)

output_details = interpreter.get_output_details()
# print(output_details)

cap = cv2.VideoCapture(video)
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
fps, fourcc = cap.get(cv2.CAP_PROP_FPS), cv2.VideoWriter_fourcc(*'mp4v')

ret, frame = cap.read()

# cv2.imshow("frame", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

model_height, model_width = input_details[0]['shape'][1], input_details[0]['shape'][2]
original_height, original_width = frame.shape[:2]

confidence_thres = 0.5
iou_thres = 0.5

class LetterBox:
    def __init__(
        self, new_shape=(model_width, model_height), auto=False, scaleFill=False, scaleup=True, center=True, stride=32
    ):
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left

    def __call__(self, labels=None, image=None):
        """Return updated labels and image with added border."""

        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img, ratio, (dw, dh)

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels."""

        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels

def preprocess(img: np.ndarray) -> np.ndarray:
    # img_height, img_width = img.shape[:2]
    
    letterbox = LetterBox(new_shape=[model_width, model_height], auto=False, stride=32)
    image, ratio, (pad_w, pad_h) = letterbox(image=img)
    image = [image]
    image = np.stack(image)
    image = image[..., :: -1].transpose((0, 3, 1, 2))
    img = np.ascontiguousarray(image, dtype=np.float32)
    
    return img / 255, ratio, (pad_w, pad_h)

def draw_detections(img, box, score, class_id):
    x1, y1, w, h = box

    # Retrieve the color for the class ID
    color = (255, 255, 0)

    # Draw the bounding box on the image
    cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

    # Create the label text with class name and score
    label = f"sidewalk: {score:.2f}"

    # Calculate the dimensions of the label text
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    # Calculate the position of the label text
    label_x = x1
    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

    # Draw a filled rectangle as the background for the label text
    cv2.rectangle(
        img,
        (int(label_x), int(label_y - label_height)),
        (int(label_x + label_width), int(label_y + label_height)),
        color,
        cv2.FILLED,
    )

    # Draw the label text on the image
    cv2.putText(img, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def scale_masks(masks, img_shape, ratio_pad=None):
    im1_shape = masks.shape[:2]
    
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_shape[0] / img_shape[0], im1_shape[1] / img_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - img_shape[1] * gain) / 2, (im1_shape[0] - img_shape[0] * gain) / 2  # wh padding
    else:
        pad = ratio_pad[1]

    # Calculate tlbr of mask
    top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
    bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
    
    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(
        masks, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR
    )  # INTER_CUBIC would be better
    
    if len(masks.shape) == 2:
        masks = masks[:, :, None]
        
    return masks

def crop_masks(masks, bboxes):
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(bboxes[:, :, None], 4, 1)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]
    c = np.arange(h, dtype=x1.dtype)[None, :, None]
    
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def process_masks(protos, masks_in, bboxes, img_shape):
    mh, mw, c = protos.shape
    
    masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)
    masks = np.ascontiguousarray(masks)
    masks = scale_masks(masks, img_shape)
    masks = np.einsum("HWN->NHW", masks)
    masks = crop_masks(masks, bboxes)
    return np.greater(masks, 0.5)

def postprocess(input_image, output, ratio, pad_w, pad_h, confidence_thres=0.5, iou_threshold=0.5, nm=32):
    """
    """
    preds, protos = output
    
    preds = np.einsum("bcn->bnc", preds)    
    preds = preds[np.amax(preds[..., 4:-nm], axis=-1) > confidence_thres]
    preds = np.c_[preds[..., :4], np.amax(preds[..., 4:-nm], axis=-1), np.argmax(preds[..., 4:-nm], axis=-1), preds[..., -nm:]]
    
    preds = preds[cv2.dnn.NMSBoxes(preds[:, :4], preds[:, 4], confidence_thres, iou_threshold)]
    
    if len(preds) > 0:
        preds[..., [0, 1]] -= preds[..., [2, 3]] / 2
        preds[..., [2, 3]] += preds[..., [0, 1]]
        
        preds[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
        preds[..., :4] /= min(ratio)

        # Bounding boxes boundary clamp
        preds[..., [0, 2]] = preds[:, [0, 2]].clip(0, input_image.shape[1])
        preds[..., [1, 3]] = preds[:, [1, 3]].clip(0, input_image.shape[0])
        
        masks = process_masks(protos[0], preds[:, 6:], preds[:, :4], input_image.shape)
        
        # check if masks[0] is purely False
        if not np.any(masks[0]):
            print("No masks found")
        else:
            print("Masks found")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    preprocessed_img, ratio, (pad_w, pad_h) = preprocess(frame)
    preprocessed_img = np.transpose(preprocessed_img, (0, 2, 3, 1))

    interpreter.set_tensor(input_details[0]['index'], preprocessed_img)
    interpreter.invoke()

    output = [interpreter.get_tensor(output_details[0]['index']), interpreter.get_tensor(output_details[1]['index'])]

    postprocessed_output = postprocess(frame, output, ratio, pad_w, pad_h, confidence_thres, iou_thres)

# cv2.imshow("frame", postprocessed_output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()