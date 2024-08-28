import ollama

# 라마3로부터 생성된 답변을 실시간으로 받기
stream = ollama.chat(
    model="llama3.1",
    messages=[{"role": "user", "content": "2023년에 K-POP 그룹인 'LE SSERAFIM'과 'ITZY' 중 누가 더 팬들에게 인기 많았을까?"}],
    stream=True
)

# 응답 결과 실시간 출력
for chunk in stream:
    print(chunk.get('message', {}).get('content', ''), end='', flush=True)
