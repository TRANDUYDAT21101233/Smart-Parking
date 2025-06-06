from ultralytics import SAM
import cv2

# Load a model
model = SAM("sam2.1_b.pt")

# Display model information (optional)
model.info()

input_image = "dataset/dataset/car/images/val/car_10006.png"
# Run inference

results = model(input_image, points= [[183, 551], [333, 560], [460, 600], [704, 551], [819, 302], [1067, 588]])

image = results[0].plot(labels=False)

for i, res in enumerate(results):
    nor_bboxes = res.boxes.xywhn
    with open(input_image.replace(".png", ".txt"), "w", encoding="utf-8") as f:
        for bbox in nor_bboxes:
            x, y, w, h = bbox
            f.write("0 {} {} {} {}".format(x, y, w, h) + "\n")

cv2.imshow("image", image)
cv2.waitKey(0)


