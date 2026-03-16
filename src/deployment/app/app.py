import streamlit as st
import requests
from datetime import datetime

API_URL = "http://localhost:8000/chat"

st.set_page_config(
    page_title="Movie RAG system",
    page_icon="🎬",
    layout="centered"
)

st.title("Movie Recommendation System")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("You can watch..."):
    
    # Добавляем сообщение пользователя
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Показываем индикатор "думает"
    with st.chat_message("assistant"):
        with st.spinner("Searching for films..."):
            try:
                response = requests.post(API_URL, json={"query": prompt}, timeout=30)
                
                if response.status_code == 200:
                    answer = response.json() if isinstance(response.json(), str) else response.text
                else:
                    answer = f"Server error: {response.status_code}"
                    
            except requests.exceptions.RequestException as e:
                answer = f"Server is unreachable: {str(e)}"

        st.markdown(answer)
    
    # Сохраняем ответ ассистента
    st.session_state.messages.append({"role": "assistant", "content": answer})

# ====================== КНОПКИ УПРАВЛЕНИЯ ======================
col1, col2 = st.columns(2)

with col1:
    if st.button("Remove chat history"):
        st.session_state.messages = []
        st.rerun()
with col2:
    if st.button("Reload RAG system"):
        try:
            requests.get("http://localhost:8000/health")
            st.success("RAG system reloaded")
        except:
            st.error("Unable to reload RAG system")
