import ollama
import streamlit as st

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama


st.set_page_config(page_title="로컬 LLM 챗봇", page_icon="🦙")


# --------------------------------------------------
# 1. 설치된 Ollama 모델 불러오기
# --------------------------------------------------

# 모델에 함께 보낼 최대 이전 메시지 수 (질문/답변 합산)
MAX_HISTORY = 10


def get_installed_models():
    """Ollama에 설치된 채팅용 모델 이름 목록 (임베딩 모델 제외)"""
    models = [m.model for m in ollama.list().models]
    return [name for name in models if "embed" not in name]


try:
    MODEL_OPTIONS = get_installed_models()
except Exception:
    st.error(
        "Ollama 서버에 연결할 수 없습니다.\n\n"
        "Ollama 앱이 실행 중인지 확인하세요. (터미널에서 `ollama list`)"
    )
    st.stop()

if not MODEL_OPTIONS:
    st.error("설치된 모델이 없습니다. 터미널에서 `ollama pull qwen3:4b` 등으로 모델을 받아주세요.")
    st.stop()


# --------------------------------------------------
# 2. Streamlit 화면
# --------------------------------------------------

st.title("🦙 로컬 LLM 챗봇")
st.caption("내 컴퓨터에서 돌아가는 Ollama 모델과 대화해보세요.")


# 사이드바: 모델 선택 + 대화 초기화
selected_model = st.sidebar.selectbox(
    "모델 선택",
    MODEL_OPTIONS
)

if st.sidebar.button("대화 초기화"):
    st.session_state.messages = []
    st.rerun()


# --------------------------------------------------
# 3. System Prompt
# --------------------------------------------------

SYSTEM_PROMPT = """
너는 친절한 한국어 AI 어시스턴트야.
답변은 간결하게 해줘.
"""


# --------------------------------------------------
# 4. 대화 기록 초기화
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# 5. 기존 대화 화면에 출력
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        # AI 응답이면 사용한 모델도 표시
        if message["role"] == "assistant" and "model" in message:
            st.caption(f"모델: {message['model']}")

        st.markdown(message["content"])


# --------------------------------------------------
# 6. LangChain Prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])


# --------------------------------------------------
# 7. 선택한 Ollama 모델 연결
# --------------------------------------------------

# qwen3 같은 사고(thinking) 모델은 사고 과정을 답변과 분리해서 받아야
# 화면에 <think> ... </think>가 섞여 나오지 않음
supports_thinking = "thinking" in (ollama.show(selected_model).capabilities or [])

llm = ChatOllama(
    model=selected_model,
    temperature=0,
    reasoning=True if supports_thinking else None,
)


# Prompt → LLM → 문자열 출력
chain = prompt | llm | StrOutputParser()


# --------------------------------------------------
# 8. 사용자 입력
# --------------------------------------------------

user_input = st.chat_input("메시지를 입력하세요")


if user_input:

    # 사용자 질문 표시
    with st.chat_message("user"):
        st.markdown(user_input)


    # 최근 대화만 모델에 전달 (컨텍스트 길이 초과 방지)
    history = [
        (m["role"], m["content"])
        for m in st.session_state.messages[-MAX_HISTORY:]
    ]


    # 모델 답변
    with st.chat_message("assistant"):

        st.caption(f"모델: {selected_model}")

        try:
            answer = st.write_stream(
                chain.stream({
                    "history": history,
                    "input": user_input
                })
            )
        except Exception as e:
            st.error(
                f"모델 호출에 실패했습니다: {e}\n\n"
                "Ollama가 실행 중인지, 모델이 설치돼 있는지 확인하세요."
            )
            st.stop()


    # 답변까지 성공했을 때만 질문/답변을 함께 저장
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "model": selected_model
    })
