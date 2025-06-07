import cv2



source = 'demo/parking_management.avi'
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read camera frame")
        break

    cv2.imwrite('image/demo.jpg', frame)
    break
cap.release()
cv2.destroyAllWindows()