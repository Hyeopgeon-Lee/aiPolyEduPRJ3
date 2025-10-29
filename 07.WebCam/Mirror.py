import cv2

# 웹캠 켜기
cap = cv2.VideoCapture(0)

print("거울 모드 실행 중... (종료: ESC 키)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 화면을 좌우 반전시켜 거울처럼 보이게 함
    mirror = cv2.flip(frame, 1)

    # 원본과 거울모드 화면을 나란히 보여주기
    combined = cv2.hconcat([frame, mirror])

    # 화면 표시
    cv2.imshow('Mirror Mode (Left: Original / Right: Mirror)', combined)

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) > 0:
        print("프로그램을 종료합니다.")
        break

# 웹캠 종료 및 창 닫기
cap.release()
cv2.destroyAllWindows()
