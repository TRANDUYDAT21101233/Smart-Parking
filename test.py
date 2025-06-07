import cv2
from ultralytics import YOLO
import torch
import torchvision

print(torch.__version__)
print(torch.cuda.is_available())
print(torchvision.__version__)


model = YOLO('runs/train/weights/best.pt')

source = 'Parking_Lot.mp4'
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read camera frame")
        break

    results = model(frame)
    for result in results:
        bboxes = result.boxes.xyxy
        confs = result.boxes.conf
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]

        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox.int().tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            confidence = confs[i]
            class_name = names[i]

            text = f'{class_name} {confidence:.2f}'

            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



