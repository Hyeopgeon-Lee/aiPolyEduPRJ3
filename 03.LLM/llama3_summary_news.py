import ollama  # LLaMA3.1 모델 활용
import requests  # 네이버 뉴스로부터 웹크롤링하기 위해 request 활용
from bs4 import BeautifulSoup  # 네이버 뉴스 HTML 소스의 내용을 가져오기 위해 활용

# 네이버는 정상적인 접근(사용자 직접 네이버 접속)이 아닌 웹 크롤링 등 비정상적인 접근을 방지 하게 위해 HTTP 해더 정보를 요구함
# HTTP 해더는 웹URL에 접속할 때, 네이버 서버에 제공하는 메타 정보
# 네이버는 HTTP 해더 정보 중 User-Agent 값의 유무를 체크함
# 따라서 네이버에 임의의 값을 넣어줌
headers = {
    "User-Agent": "Seoul Gangseo Campus of Korea Polytechnics College, Dept. of Data Analysis / Python Education"}

# 수집할 신문기사 URL
webpage = requests.get("https://n.news.naver.com/mnews/article/014/0005233982", headers=headers)

# URL로부터 읽은 HTML 내용을 파이썬에서 처리할 수 있게 파싱하기
soup = BeautifulSoup(webpage.content, "html.parser")

# 신문기사 본문 내용을 문자열로 저장하기
naver_news = soup.select_one("#dic_area").get_text().strip()
# naver_news = soup.select_one("#articeBody").get_text().strip()

# 신문기사 출력
print("<신문기사 원문>")
print(naver_news)

# LLaMA 3.1 모델을 사용하여 텍스트 요약
response = ollama.chat(
    model="llama3.1",
    messages=[{"role": "user", "content": f"다음 텍스트를 요약해줘:\n\n{naver_news}"}]
)

# 응답 내용에서 요약된 텍스트를 추출합니다.
# 응답이 리스트 형식으로 반환될 수 있으므로 올바르게 처리합니다.
if isinstance(response, list) and len(response) > 0:
    summary = response[0]['message']['content']
else:
    summary = response['message']['content']

# 요약 결과 출력
print("<신문기사 요약>")
print(summary)
