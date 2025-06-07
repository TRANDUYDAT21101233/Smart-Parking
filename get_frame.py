import cv2



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

    cv2.imwrite('image/parking_lot.jpg', frame)
    break
cap.release()
cv2.destroyAllWindows()