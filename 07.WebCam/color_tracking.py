import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# 추적할 색상 (HSV 기준: 파란색)
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 색상 영역 추출
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Original", frame)
    cv2.imshow("Color Tracking", result)

    if cv2.waitKey(1) > 0:
        break

cap.release()
cv2.destroyAllWindows()
