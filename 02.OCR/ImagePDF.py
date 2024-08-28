import pytesseract
import os

# Tesseract 실행 파일 경로 설정
pytesseract.pytesseract.tesseract_cmd = r"C:\Tesseract-OCR\tesseract.exe"

# Tesseract에서 사용 가능한 언어 학습 모델 확인
available_languages = pytesseract.get_languages()
print("인식 가능한 언어(저장되어 있는 언어별 학습모델 파일):", available_languages)

# 이미지 파일 경로 및 PDF 저장 경로 설정
image_path = "../image/news01.jpg"
output_pdf_path = "../pdf/news01.pdf"

# 이미지 파일이 존재하는지 확인
if not os.path.exists(image_path):
    raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

# 이미지에서 텍스트를 인식하여 PDF로 변환
pdf = pytesseract.image_to_pdf_or_hocr(image_path, extension="pdf")

# PDF 파일 저장 경로의 디렉토리가 존재하지 않으면 생성
output_dir = os.path.dirname(output_pdf_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 변환된 PDF 파일 저장
f = open(output_pdf_path, "w+b")
f.write(pdf)
f.close()

print("PDF 파일 생성이 완료되었습니다:", output_pdf_path)
