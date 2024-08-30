import cv2

image = cv2.imread("../image/my_face.jpg", cv2.IMREAD_COLOR)  # 지정된 경로에서 이미지를 컬러로 불러옴

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # RGB 이미지를 흑백 이미지로 변경

gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))  # 흑백 이미지를 히스토그램 평활화를 적용

# 얼굴 검출기를 로드 및 얼굴 검출 수행
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")  # 하르 기반 얼굴 검출기를 로드

# 얼굴 검출
# gray : 히스토그램 변환된 이미지에서 얼굴을 검출합니다.
# scaleFactor = 1.1 : 이미지 크기를 피라미드 방식으로 줄여가면서 얼굴을 검출합니다. 1.1은 10%씩 이미지 크기를 줄이는 것을 의미
# minNeighbors = 5 : 검출된 얼굴 영역의 이웃 수를 지정합니다. 이 값이 클수록 더 엄격하게 얼굴을 검출 수행
#                    즉, 더 많은 이웃을 가진 후보군만 최종 얼굴로 판단
# minSize = (100, 100) : 검출할 얼굴의 최소 크기를 지정합니다. 여기서는 너비와 높이가 100x100 픽셀 이상인 얼굴만 검출
faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100))

# 인식된 얼굴의 수
facesCnt = len(faces)

# 인식된 얼굴의 수 출력
print(len(faces))

# 검출된 얼굴의 수만큼 반복하여 실행함
for face in faces:
    # 얼굴 위치 값을 가져오기
    x, y, w, h = face

    # 원본이미지로부터 얼굴영역 가져오기
    face_image = image[y:y + h, x:x + w]

    # 얼굴 영역에 블러 처리 / Kernel Size가 클수록 블러가 강해짐
    blur_face_image = cv2.blur(face_image, (50, 50))

    # 원본이미지에 블러 처리한 얼굴 이미지 붙이기
    image[y:y + h, x:x + w] = blur_face_image

# 블러 처리된 이미지 파일 생성하기
cv2.imwrite("../result/face_blur.jpg", image)

# 블러 처리된 이미지 보여주기
cv2.imshow("face-blur", cv2.imread("../result/face_blur.jpg", cv2.IMREAD_COLOR))

# 입력받는 것 대기하기, 작성안하면, 결과창이 바로 닫힘
cv2.waitKey(0)
