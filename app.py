import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

# 미리 ollama pull로 받아둔 모델 중에서 고르게 합니다.
MODEL_OPTIONS = ["qwen3.5:9b", "llama3.2:latest", "mistral:latest"]
SYSTEM_PROMPT = "너는 친절한 한국어 AI 어시스턴트야. 답변은 간결하게 해줘."

st.title("🦙 로컬 LLM 챗봇")
st.caption("내 컴퓨터에서 돌아가는 Ollama 모델과 대화해보세요.")

selected_model = st.sidebar.selectbox("모델 선택", MODEL_OPTIONS)

# 대화 기록을 저장할 리스트. 새로고침해도 유지되도록 session_state에 둡니다.
if "messages" not in st.session_state:
    st.session_state.messages = []

# 지금까지의 대화를 화면에 그립니다.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# LangChain 체인: 프롬프트 조립 -> 모델 호출 -> 문자열로 변환
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])
llm = ChatOllama(model=selected_model)
chain = prompt | llm | StrOutputParser()

user_input = st.chat_input("메시지를 입력하세요")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 마지막 사용자 입력을 빼고, 그 앞까지가 "이전 대화"입니다.
    history = [(m["role"], m["content"]) for m in st.session_state.messages[:-1]]

    with st.chat_message("assistant"):
        answer = st.write_stream(chain.stream({"history": history, "input": user_input}))

    st.session_state.messages.append({"role": "assistant", "content": answer})
