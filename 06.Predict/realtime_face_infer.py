from PIL import ImageFont, ImageDraw, Image
import cv2, numpy as np, time, os

# ==== 한글 폰트 ====
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
font = ImageFont.truetype(FONT_PATH, 18)


def draw_korean_text(img_bgr, text, pos=(10, 30), color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ==== 모델/상수 ====
ONNX_PATH = "../model/saved_multihead_model.onnx"
IMG_SIZE = 128
MAX_AGE = 116.0

GENDER = {0: "남자", 1: "여자"}
RACE = {0: "백인", 1: "흑인", 2: "아시아인", 3: "인디언", 4: "기타"}

# ==== ONNX 로드 ====
net = cv2.dnn.readNetFromONNX(ONNX_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
OUTS = net.getUnconnectedOutLayersNames()

# ==== 얼굴 검출기 ====
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
)

def infer_face(face_bgr):
    # 히스토그램 평활화(밝기 개선)
    ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.equalizeHist(y)
    face_eq = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)

    # 전처리
    img = cv2.cvtColor(face_eq, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    inp = np.expand_dims(img, 0)  # (1,128,128,3)

    net.setInput(inp)
    outs = net.forward(OUTS)
    out = {OUTS[i]: outs[i] for i in range(len(OUTS))}

    def pick(name_part):
        for k in out:
            if name_part in k or name_part == k:
                return out[k]
        # 못 찾으면 첫 출력 사용(예외 방지)
        return list(out.values())[0]

    age = float(pick("age_out").reshape(-1)[0]) * MAX_AGE
    gender_id = int(np.argmax(pick("gender_out").reshape(-1)))
    race_id = int(np.argmax(pick("race_out").reshape(-1)))
    return int(round(age)), GENDER[gender_id], RACE[race_id]


# ==== 웹캠 ====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")

print("실행 중... (종료: q)")
fps, prev_t = 0.0, time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS 계산
    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - prev_t))
    prev_t = now

    # 얼굴 검출
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        if face.size == 0:
            continue

        try:
            age, gender, race = infer_face(face)
            label = f"나이 {age}세 · 성별 {gender} · 인종 {race}"
            color = (0, 255, 0)
        except Exception:
            label = "추론 오류"
            color = (0, 0, 255)

        # 박스
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        # 얼굴 상단에 한글 라벨( Pillow 사용 )
        frame = draw_korean_text(frame, label, (x, max(0, y - 30)), color)

    # 좌상단에 FPS(한글로 표시)
    frame = draw_korean_text(frame, f"FPS: {fps:.1f}", (10, 10), (255, 255, 255))

    # 창 제목은 영문(유니코드 깨짐 방지)
    cv2.imshow("Age/Gender/Race (ONNX)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

