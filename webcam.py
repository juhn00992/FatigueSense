import cv2
from face_landmarks import detect_yawn
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to grab frame")
        break

    frame = detect_yawn(frame)
    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()