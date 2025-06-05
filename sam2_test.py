from ultralytics import SAM
import cv2

# Load a model
model = SAM("sam2.1_b.pt")

# Display model information (optional)
model.info()

input_image = "image/car_10004.png"
# Run inference

results = model(input_image, points= [[499, 577], [763, 494], [845, 243], [853, 737], [1114, 701], [1380, 418], [1590, 226], [1761, 564], [1655, 567], [1495, 568], [1382, 577], [1446, 802], [1531, 799], [1589, 797], [1651, 797], [1722, 794]])


image = results[0].plot(labels=False)

for i, res in enumerate(results):
    nor_bboxes = res.boxes.xywhn
    with open(input_image.replace(".png", ".txt"), "w", encoding="utf-8") as f:
        for bbox in nor_bboxes:
            x, y, w, h = bbox
            f.write("0 {} {} {} {}".format(x, y, w, h) + "\n")

cv2.imshow("image", image)
cv2.waitKey(0)


