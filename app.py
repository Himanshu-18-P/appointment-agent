import os
import streamlit as st
from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage  # ✅ Proper single import

from tools import agent, memory  # Ensure tools.py exports agent and memory correctly

# Load environment variables
load_dotenv()

# Streamlit UI setup
st.set_page_config(page_title="DoctorBot 🤖", layout="centered")
st.title("🩺 DoctorBot Appointment Scheduler")

# Initialize session state chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="👋 Hello! I'm DoctorBot. How can I help you with appointments today?")
    ]

# Sync memory with chat history
memory.chat_memory.messages = st.session_state.chat_history

# Display chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# Chat input interface
if prompt := st.chat_input("Ask about availability, booking, etc..."):
    st.chat_message("user").write(prompt)
    st.session_state.chat_history.append(HumanMessage(content=prompt))  # Add user input

    memory.chat_memory.messages = st.session_state.chat_history  # Sync memory

    try:
        response = agent.invoke({"input": prompt})["output"]
        st.session_state.chat_history.append(AIMessage(content=response))  # Add bot response
        st.chat_message("assistant").write(response)

    except Exception as e:
        st.chat_message("assistant").error(f"❌ Error: {e}")
