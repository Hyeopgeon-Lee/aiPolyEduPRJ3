import cv2

image = cv2.imread("../image/my_face.jpg", cv2.IMREAD_COLOR)  # 지정된 경로에서 이미지를 컬러로 불러옴
emoticon_image = cv2.imread("../image/emoticon.png", cv2.IMREAD_COLOR)

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
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

print(f"검출된 얼굴 수: {len(faces)}")  # 검출된 얼굴의 수를 출력

for (x, y, w, h) in faces:
    # 얼굴 영역 크기에 맞게 이모티콘 이미지 크기 조정
    resized_emoticon = cv2.resize(emoticon_image, (w, h), interpolation=cv2.INTER_AREA)

    # 이모티콘 이미지를 그레이스케일로 변환 및 마스크 생성
    emoticon_gray = cv2.cvtColor(resized_emoticon, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(emoticon_gray, 240, 255, cv2.THRESH_BINARY_INV)

    # 마스크 반전 생성
    mask_inv = cv2.bitwise_not(mask)

    # 이모티콘에서 이모티콘 부분만 추출
    emoticon_fg = cv2.bitwise_and(resized_emoticon, resized_emoticon, mask=mask)

    # 원본 이미지에서 얼굴 부분만 추출
    face_bg = cv2.bitwise_and(image[y:y + h, x:x + w], image[y:y + h, x:x + w], mask=mask_inv)

    # 얼굴 부분에 이모티콘 합성
    image[y:y + h, x:x + w] = cv2.add(face_bg, emoticon_fg)

# 이모티콘 처리된 이미지 저장 및 표시
result_path = "../result/emoticon_result.jpg"
cv2.imwrite(result_path, image)
cv2.imshow("Emoticon Applied Image", image)

# 입력 대기 (결과 창이 바로 닫히지 않도록 설정)
cv2.waitKey(0)
cv2.destroyAllWindows()
