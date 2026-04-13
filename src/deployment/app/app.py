import os

import streamlit as st
import requests
from datetime import datetime

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
CHAT_URL = f"{API_BASE}/chat"
RELOAD_URL = f"{API_BASE}/reload"

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
                response = requests.post(CHAT_URL, json={"query": prompt}, timeout=300)
                
                if response.status_code == 200:
                    payload = response.json()
                    if isinstance(payload, dict) and "recommendation" in payload:
                        answer = payload["recommendation"]
                    else:
                        answer = str(payload)
                else:
                    answer = f"Server error: {response.status_code} ({response.text})"
                    
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
            r = requests.post(RELOAD_URL, timeout=300)
            if r.status_code == 200:
                st.success("RAG system reloaded")
            else:
                st.error(f"Unable to reload RAG system: {r.status_code} ({r.text})")
        except Exception as e:
            st.error(f"Unable to reload RAG system: {e}")
