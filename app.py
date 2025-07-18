import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

MODEL_NAME = "SEA-LION 3.5 8B"

if MODEL_NAME == "Cendol 7B":
    from cendol_7b.graph import run_agent
elif MODEL_NAME == "SahabatAI Gemma2 9B":
    from sahabatai_gemma2_9b.graph import run_agent
elif MODEL_NAME == "Qwen 2.5 7B":
    from qwen_25_7b.graph import run_agent
elif MODEL_NAME == "SEA-LION 3.5 8B":
    from sealion_35_8b.graph import run_agent
else:
    raise ValueError(f"Model {MODEL_NAME} not found")
    
st.set_page_config(page_title="🤖 Legal AI Assistant", layout="centered")
st.title("🤖 Legal AI Assistant")

def get_response(messages):
    formatted_messages = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    result = run_agent(formatted_messages)
    
    return result["final_answer"]

if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Halo! Saya adalah asisten AI legal yang bisa membantu Anda dalam memahami peraturan perundang-undangan di Indonesia.")
    ]

for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

if prompt := st.chat_input("What's on your mind?"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("human"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        ai_response_content = get_response(st.session_state.messages)
        ai_response = AIMessage(content=ai_response_content)
        st.session_state.messages.append(ai_response)
        with st.chat_message("ai"):
            st.markdown(ai_response_content)