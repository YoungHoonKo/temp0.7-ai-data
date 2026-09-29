# 로컬 LLM 챗봇 데모 (Ollama + LangChain + Streamlit)

동아리 부원들에게 "로컬에서 돌아가는 LLM"과 "LangChain으로 프롬프트를 조립하는 흐름"을
직접 체험시켜주기 위한 최소 예제입니다.

## 사전 준비

1. [Ollama](https://ollama.com) 설치
2. 모델 하나 이상 받아두기 (터미널에서):
   ```bash
   ollama pull llama3.2
   ```
3. Ollama 서버가 켜져 있어야 합니다 (맥에서는 앱 실행 시 자동으로 백그라운드 실행됩니다).
   확인:
   ```bash
   ollama list
   ```

## 실행 방법

```bash
cd ollama-chatbot-demo
python3 -m venv venv
source venv/bin/activate      # Windows는 venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

브라우저가 자동으로 열리고, 사이드바에서 모델을 고른 뒤 채팅창에 메시지를 입력하면 됩니다.

## 코드 구조 살펴보기 (LangChain 핵심 개념)

`app.py` 하나로 구성되어 있고, 흐름은 다음과 같습니다.

- `ChatPromptTemplate` : 시스템 프롬프트 + 이전 대화 기록 + 새 질문을 하나의
  프롬프트로 조립하는 템플릿입니다.
- `ChatOllama` : LangChain에서 로컬 Ollama 모델을 호출하는 래퍼(wrapper)입니다.
- `prompt | llm | StrOutputParser()` : `|` 연산자로 이어붙인 것이 LangChain의
  **체인(chain)** 입니다. "프롬프트 조립 → 모델 호출 → 문자열로 파싱"까지 한 줄로 표현됩니다.
- `chain.stream(...)` : 답변을 토큰 단위로 스트리밍해서, ChatGPT처럼 글자가
  하나씩 타이핑되는 효과를 냅니다.

이 구조를 이해하면 이후에 RAG(문서 검색 연동), 도구 호출(tool calling) 같은
더 복잡한 LangChain 예제로 자연스럽게 확장할 수 있습니다.
