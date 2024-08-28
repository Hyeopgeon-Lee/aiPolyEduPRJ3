import cv2

# 분석할 이미지 불러오기
image_path = "../image/my_face.jpg"  # 이미지 파일 경로를 지정
image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 지정된 경로에서 이미지를 컬러로 불러옴
if image is None:  # 이미지가 정상적으로 불러와지지 않았을 경우를 확인합니다.
    raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")  # 이미지가 없으면 오류를 발생시킴

# 이미지를 그레이스케일로 변환 및 히스토그램 평활화 적용
gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))  # 이미지를 그레이스케일로 변환하고 히스토그램 평활화를 적용합

# 얼굴 검출기를 로드 및 얼굴 검출 수행
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")  # 하르 기반 얼굴 검출기를 로드

# 얼굴 검출
# gray : 히스토그램 변환된 이미지에서 얼굴을 검출합니다.
# scaleFactor = 1.1 : 이미지 크기를 피라미드 방식으로 줄여가면서 얼굴을 검출합니다. 1.1은 10%씩 이미지 크기를 줄이는 것을 의미
# minNeighbors = 5 : 검출된 얼굴 영역의 이웃 수를 지정합니다. 이 값이 클수록 더 엄격하게 얼굴을 검출 수행
#                    즉, 더 많은 이웃을 가진 후보군만 최종 얼굴로 판단
# minSize = (100, 100) : 검출할 얼굴의 최소 크기를 지정합니다. 여기서는 너비와 높이가 100x100 픽셀 이상인 얼굴만 검출
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

print(f"검출된 얼굴 수: {len(faces)}")  # 검출된 얼굴의 수를 출력

for (x, y, w, h) in faces:  # 검출된 얼굴 각각에 대해 반복합니다.
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 얼굴 주위에 파란색 사각형 그리기

# 결과 이미지 출력
cv2.imshow("Detected Faces", image)  # 결과 이미지를 화면에 표시
cv2.waitKey(0)  # 키 입력을 기다리기
cv2.destroyAllWindows()  # 모든 창을 닫기
