import ollama

# 텍스트 파일 경로 지정
file_path = "../text/news.txt"

# 텍스트 파일 내용을 읽습니다.
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# LLaMA 3.1 모델을 사용하여 텍스트 요약
response = ollama.chat(
    model="llama3.1",
    messages=[{"role": "user", "content": f"다음 텍스트를 요약해줘:\n\n{text}"}]
)

# 응답 내용에서 요약된 텍스트를 추출합니다.
# 응답이 리스트 형식으로 반환될 수 있으므로 올바르게 처리합니다.
if isinstance(response, list) and len(response) > 0:
    summary = response[0]['message']['content']
else:
    summary = response['message']['content']

# 요약 결과 출력
print("<요약 결과>")
print(summary)
