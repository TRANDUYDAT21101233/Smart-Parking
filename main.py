import cv2
from ultralytics import solutions
import torch
import torchvision

print(torch.__version__)
print(torch.cuda.is_available())
print(torchvision.__version__)

# Video capture
cap = cv2.VideoCapture("Parking_Lot.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("parking management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize parking management object
parking_manager = solutions.ParkingManagement(
    model="runs/train/weights/best.pt",  # path to model file
    json_file="bounding_boxes.json",  # path to parking annotations file
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = parking_manager(frame)

    # print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows