import ollama

stream = ollama.chat(  # 라마3로부터 생성된 답변을 실시간 받기
    # 사용 모델
    model="llama3.1",

    # 명령어
    messages=[{"role": "user", "content": "2023년에 K-POP 그룹인 'LE SSERAFIM'과  'ITZY' 중 누가 더 팬들에게 인기 많았을까?"}],

    # 응답결과 실시간 받기 설정
    stream=True,
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)  # 응답 결과 실시간 출력하기

