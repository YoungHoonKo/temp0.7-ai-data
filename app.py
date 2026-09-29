import streamlit as st

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama


# --------------------------------------------------
# 1. 사용할 Ollama 모델
# --------------------------------------------------

MODEL_OPTIONS = [
    "qwen3:4b",
    "qwen3:8b",
]


# --------------------------------------------------
# 2. Streamlit 화면
# --------------------------------------------------

st.title("🦙 로컬 LLM 챗봇")
st.caption("내 컴퓨터에서 돌아가는 Ollama 모델과 대화해보세요.")


# 사이드바에서 모델 선택
selected_model = st.sidebar.selectbox(
    "모델 선택",
    MODEL_OPTIONS
)


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

llm = ChatOllama(
    model=selected_model,
    temperature=0,
)


# Prompt → LLM → 문자열 출력
chain = prompt | llm | StrOutputParser()


# --------------------------------------------------
# 8. 사용자 입력
# --------------------------------------------------

user_input = st.chat_input("메시지를 입력하세요")


if user_input:

    # 사용자 질문 저장
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })


    # 사용자 질문 표시
    with st.chat_message("user"):
        st.markdown(user_input)


    # 마지막 사용자 입력을 제외한 이전 대화
    history = [
        (m["role"], m["content"])
        for m in st.session_state.messages[:-1]
    ]


    # 모델 답변
    with st.chat_message("assistant"):

        st.caption(f"모델: {selected_model}")

        answer = st.write_stream(
            chain.stream({
                "history": history,
                "input": user_input
            })
        )


    # 모델 이름까지 함께 저장
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "model": selected_model
    })