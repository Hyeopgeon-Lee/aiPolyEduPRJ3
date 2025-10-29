import cv2
import numpy as np
import datetime
import os

# 저장할 폴더 이름
CAPTURE_DIR = "capture"

# 'capture' 폴더가 없으면 새로 만든다
if not os.path.exists(CAPTURE_DIR):
    os.makedirs(CAPTURE_DIR)

# 웹캠 켜기
cap = cv2.VideoCapture(0)

# 처음 두 개의 화면(프레임)을 읽는다
ret, frame1 = cap.read()
ret, frame2 = cap.read()

print("모션 감지를 시작합니다. (종료: ESC 키)")

while cap.isOpened():
    # 두 화면(프레임)의 차이 계산 → 움직임 확인용
    diff = cv2.absdiff(frame1, frame2)

    # 색을 흑백으로 바꾼다 (계산 단순화)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # 작은 노이즈(잡음)를 없애서 깔끔하게 만든다
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 차이가 큰(움직임이 있는) 부분만 남긴다
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # 흰 부분(움직임 영역)을 조금 더 키워준다
    dilated = cv2.dilate(thresh, None, iterations=3)

    # 움직임의 윤곽선(테두리)을 찾는다
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    for contour in contours:
        # 너무 작은 움직임은 무시한다
        if cv2.contourArea(contour) < 2000:
            continue

        # 움직인 부분에 사각형 표시
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        motion_detected = True

    # 움직임이 감지된 경우
    if motion_detected:
        # 화면에 글씨 표시
        cv2.putText(frame1, "Motion Detected!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 현재 시각으로 파일 이름 만들기
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(CAPTURE_DIR, f"capture_{timestamp}.jpg")

        # 화면(이미지)을 파일로 저장
        cv2.imwrite(filename, frame1)

        # 콘솔에 메시지 출력
        print(f"움직임 감지됨: {filename}")

    # 화면에 현재 영상 보여주기
    cv2.imshow("Motion Detection", frame1)

    # 다음 비교를 위해 프레임 교체
    frame1 = frame2
    ret, frame2 = cap.read()

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) > 0:
        print("프로그램을 종료합니다.")
        break

# 웹캠과 창 닫기
cap.release()
cv2.destroyAllWindows()
