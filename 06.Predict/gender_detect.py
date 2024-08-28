import cv2

# 얼굴 탐지를 위한 Haar Cascade 모델 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

# 성별 예측을 위한 모델 로드 (Caffe 모델)
gender_net = cv2.dnn.readNetFromCaffe(
    "../model/deploy_gender.prototxt", "../model/gender_net.caffemodel")

# 모델에서 사용하는 평균값 (RGB 순서)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# 성별 예측 결과에 대한 리스트
gender_list = ["Male", "Female"]

# 분석할 이미지 로드
image = cv2.imread("../image/my_face.jpg")

# 이미지가 정상적으로 로드되지 않은 경우 예외 처리
if image is None:
    raise Exception("이미지를 불러올 수 없습니다.")

# 이미지를 그레이스케일로 변환 및 히스토그램 평활화
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

# 얼굴 검출 수행
faces = face_cascade.detectMultiScale(
    gray, scaleFactor=1.5, minNeighbors=5, minSize=(100, 100))

# 탐지된 얼굴 수만큼 반복하여 성별 예측 수행
for (x, y, w, h) in faces:
    # 얼굴 영역 추출
    face_image = image[y:y + h, x:x + w]

    # 네트워크 입력을 위한 BLOB 생성
    blob = cv2.dnn.blobFromImage(
        face_image, scalefactor=1.0, size=(227, 227), mean=MODEL_MEAN_VALUES, swapRB=False)

    # 성별 예측 수행
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_preds.argmax()  # 예측 결과에서 가장 높은 확률을 가진 성별 선택

    # 얼굴 영역에 사각형 그리기
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 4)

    # 성별 예측 결과를 이미지에 추가
    result_text = f"Gender: {gender_list[gender]}"
    cv2.putText(image, result_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# 결과 이미지 출력
cv2.imshow("Gender Prediction", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
