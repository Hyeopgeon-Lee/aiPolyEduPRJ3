import ollama
import pyttsx3

# pyttsx3 엔진 초기화
engine = pyttsx3.init()

# 라마3로부터 생성된 답변을 실시간으로 받기
stream = ollama.chat(
    model="llama3.1",
    messages=[{"role": "user", "content": "2023년에 K-POP 그룹인 'LE SSERAFIM'과 'ITZY' 중 누가 더 팬들에게 인기 많았을까?"}],
    stream=True
)

# 응답 결과 실시간 출력 및 음성으로 읽어주기
for chunk in stream:
    content = chunk.get('message', {}).get('content', '')
    if content:  # 내용이 있을 때만 출력 및 읽기
        print(content, end='', flush=True)
        engine.say(content)  # 음성으로 읽기
        engine.runAndWait()  # 음성 출력 대기
