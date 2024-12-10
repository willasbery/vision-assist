class Results(SimpleClass):
    """
    A class for storing and manipulating inference results.

    This class encapsulates the functionality for handling detection, segmentation, pose estimation,
    and classification results from YOLO models.

    Attributes:
        orig_img (numpy.ndarray): Original image as a numpy array.
        orig_shape (Tuple[int, int]): Original image shape in (height, width) format.
        boxes (Boxes | None): Object containing detection bounding boxes.
        masks (Masks | None): Object containing detection masks.
        probs (Probs | None): Object containing class probabilities for classification tasks.
        keypoints (Keypoints | None): Object containing detected keypoints for each object.
        obb (OBB | None): Object containing oriented bounding boxes.
        speed (Dict[str, float | None]): Dictionary of preprocess, inference, and postprocess speeds.
        names (Dict[int, str]): Dictionary mapping class IDs to class names.
        path (str): Path to the image file.
        _keys (Tuple[str, ...]): Tuple of attribute names for internal use.

    Methods:
        update: Updates object attributes with new detection results.
        cpu: Returns a copy of the Results object with all tensors on CPU memory.
        numpy: Returns a copy of the Results object with all tensors as numpy arrays.
        cuda: Returns a copy of the Results object with all tensors on GPU memory.
        to: Returns a copy of the Results object with tensors on a specified device and dtype.
        new: Returns a new Results object with the same image, path, and names.
        plot: Plots detection results on an input image, returning an annotated image.
        show: Shows annotated results on screen.
        save: Saves annotated results to file.
        verbose: Returns a log string for each task, detailing detections and classifications.
        save_txt: Saves detection results to a text file.
        save_crop: Saves cropped detection images.
        tojson: Converts detection results to JSON format.

    Examples:
        >>> results = model("path/to/image.jpg")
        >>> for result in results:
        ...     print(result.boxes)  # Print detection boxes
        ...     result.show()  # Display the annotated image
        ...     result.save(filename="result.jpg")  # Save annotated image
    """

    def __init__(
        self, orig_img, path, names, boxes=None, masks=None, probs=None, keypoints=None, obb=None, speed=None
    ) -> None:
        """
        Initialize the Results class for storing and manipulating inference results.

        Args:
            orig_img (numpy.ndarray): The original image as a numpy array.
            path (str): The path to the image file.
            names (Dict): A dictionary of class names.
            boxes (torch.Tensor | None): A 2D tensor of bounding box coordinates for each detection.
            masks (torch.Tensor | None): A 3D tensor of detection masks, where each mask is a binary image.
            probs (torch.Tensor | None): A 1D tensor of probabilities of each class for classification task.
            keypoints (torch.Tensor | None): A 2D tensor of keypoint coordinates for each detection.
            obb (torch.Tensor | None): A 2D tensor of oriented bounding box coordinates for each detection.
            speed (Dict | None): A dictionary containing preprocess, inference, and postprocess speeds (ms/image).

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> result = results[0]  # Get the first result
            >>> boxes = result.boxes  # Get the boxes for the first result
            >>> masks = result.masks  # Get the masks for the first result

        Notes:
            For the default pose model, keypoint indices for human body pose estimation are:
            0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
            5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow
            9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip
            13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle
        """
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None  # native size boxes
        self.masks = Masks(masks, self.orig_shape) if masks is not None else None  # native size or imgsz masks
        self.probs = Probs(probs) if probs is not None else None
        self.keypoints = Keypoints(keypoints, self.orig_shape) if keypoints is not None else None
        self.obb = OBB(obb, self.orig_shape) if obb is not None else None
        self.speed = speed if speed is not None else {"preprocess": None, "inference": None, "postprocess": None}
        self.names = names
        self.path = path
        self.save_dir = None
        self._keys = "boxes", "masks", "probs", "keypoints", "obb"

    def __getitem__(self, idx):
        """
        Return a Results object for a specific index of inference results.

        Args:
            idx (int | slice): Index or slice to retrieve from the Results object.

        Returns:
            (Results): A new Results object containing the specified subset of inference results.

        Examples:
            >>> results = model("path/to/image.jpg")  # Perform inference
            >>> single_result = results[0]  # Get the first result
            >>> subset_results = results[1:4]  # Get a slice of results
        """
        return self._apply("__getitem__", idx)

    def __len__(self):
        """
        Return the number of detections in the Results object.

        Returns:
            (int): The number of detections, determined by the length of the first non-empty attribute
                (boxes, masks, probs, keypoints, or obb).

        Examples:
            >>> results = Results(orig_img, path, names, boxes=torch.rand(5, 4))
            >>> len(results)
            5
        """
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                return len(v)

    def update(self, boxes=None, masks=None, probs=None, obb=None):
        """
        Updates the Results object with new detection data.

        This method allows updating the boxes, masks, probabilities, and oriented bounding boxes (OBB) of the
        Results object. It ensures that boxes are clipped to the original image shape.

        Args:
            boxes (torch.Tensor | None): A tensor of shape (N, 6) containing bounding box coordinates and
                confidence scores. The format is (x1, y1, x2, y2, conf, class).
            masks (torch.Tensor | None): A tensor of shape (N, H, W) containing segmentation masks.
            probs (torch.Tensor | None): A tensor of shape (num_classes,) containing class probabilities.
            obb (torch.Tensor | None): A tensor of shape (N, 5) containing oriented bounding box coordinates.

        Examples:
            >>> results = model("image.jpg")
            >>> new_boxes = torch.tensor([[100, 100, 200, 200, 0.9, 0]])
            >>> results[0].update(boxes=new_boxes)
        """
        if boxes is not None:
            self.boxes = Boxes(ops.clip_boxes(boxes, self.orig_shape), self.orig_shape)
        if masks is not None:
            self.masks = Masks(masks, self.orig_shape)
        if probs is not None:
            self.probs = probs
        if obb is not None:
            self.obb = OBB(obb, self.orig_shape)