import cv2, os, time, datetime

CAPTURE_DIR = "capture"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# --- 파라미터(필요시 숫자만 조정) ---
MIN_CAPTURE_INTERVAL = 3.0   # 초: 캡처 최소 간격(쿨다운)
REQUIRED_CONSEC_FRAMES = 3   # 연속으로 N프레임 이상 감지 시 캡처
THRESH = 25                  # 이진화 임계값(작을수록 민감)
DILATE_ITER = 2              # 팽창 반복 횟수(더 크면 덩어리화)
BLUR_KSIZE = 5               # 가우시안 커널(홀수)
# ------------------------------------

cap = cv2.VideoCapture(0)
ok, prev = cap.read()
ok2, curr = cap.read()
if not (ok and ok2):
    raise SystemExit("카메라 프레임을 읽을 수 없습니다.")

motion_streak = 0
last_capture_ts = 0.0

print("모션 감지 시작 (종료: ESC)")

while True:
    # 차분 → 그레이 → 블러 → 이진화 → 팽창
    diff  = cv2.absdiff(prev, curr)
    gray  = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
    _, th = cv2.threshold(blur, THRESH, 255, cv2.THRESH_BINARY)
    th    = cv2.dilate(th, None, iterations=DILATE_ITER)

    # 화면 크기에 비례한 최소 면적(0.5% 권장)
    h, w = th.shape
    min_area = int(w * h * 0.005)   # 0.5% ; 과하면 0.003~0.008로 조절

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion = False
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, ww, hh = cv2.boundingRect(c)
        cv2.rectangle(curr, (x, y), (x+ww, y+hh), (0, 255, 0), 2)
        motion = True

    # 연속 감지 카운트
    motion_streak = motion_streak + 1 if motion else 0

    # 캡처 조건: 연속 N프레임 이상 & 쿨다운 지남
    now = time.time()
    if motion_streak >= REQUIRED_CONSEC_FRAMES and (now - last_capture_ts) >= MIN_CAPTURE_INTERVAL:
        cv2.putText(curr, "Motion Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(CAPTURE_DIR, f"capture_{ts}.jpg")
        cv2.imwrite(path, curr)
        last_capture_ts = now
        print("캡처:", path)

    cv2.imshow("Motion Detection", curr)

    # 다음 비교 준비
    prev = curr
    ok, curr = cap.read()
    if not ok:
        break

    # ESC 로 종료
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
