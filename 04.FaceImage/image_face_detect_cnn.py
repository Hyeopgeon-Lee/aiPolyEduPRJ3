import cv2  # OpenCV 라이브러리
import dlib  # Dlib 라이브러리

# CNN 기반 얼굴 검출기 모델 로드
cnn_face_detector = dlib.cnn_face_detection_model_v1("../model/mmod_human_face_detector.dat")

# 분석할 이미지 불러오기
image_path = "../image/my_face.jpg"  # 이미지 파일 경로를 지정
image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 지정된 경로에서 이미지를 컬러로 불러옴

if image is None:  # 이미지가 정상적으로 불러와지지 않았을 경우를 확인합니다.
    raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")  # 이미지가 없으면 오류를 발생시킴

if image is None:  # 이미지가 정상적으로 불러와지지 않았을 경우를 확인합니다.
    raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image}")  # 이미지가 없으면 오류를 발생시킴

# 이미지를 RGB로 변환 (dlib은 RGB 이미지를 필요로 함) / 흑백 및 히스토그램 평활화 필요없음
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# CNN 모델을 사용해 얼굴 검출 수행
faces = cnn_face_detector(rgb_image)

# 검출된 얼굴에 대해 사각형 그리기
for face in faces:  # 검출된 각 얼굴에 대해 반복 작업을 수행합니다.
    # dlib의 CNN 모델은 확률과 좌표를 함께 반환함
    x, y, w, h = (face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height())

    # 얼굴 영역에 사각형 그리기
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# 결과 이미지 출력
cv2.imshow("CNN Face Detection", image)  # 결과 이미지를 새로운 창에서 표시합니다.
cv2.waitKey(0)  # 사용자가 키를 누를 때까지 창을 유지합니다.
cv2.destroyAllWindows()  # 모든 창을 닫고 리소스를 해제합니다.
