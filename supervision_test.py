import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
image = cv2.imread("image/car.png")

def callback(image_slice: np.ndarray) -> sv.Detections:
    result = model(image_slice)[0]
    return sv.Detections.from_ultralytics(result)

slicer = sv.InferenceSlicer(callback = callback)
detections = slicer(image)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

cv2.imwrite("supervision_output.png", annotated_image)