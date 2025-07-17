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
    def run_agent(messages):
        return {"final_answer": "TestMessage"}
    
st.set_page_config(page_title="🤖 Chat Dashboard", layout="centered")
st.title("🤖 Chat Dashboard")

# Function to get response from Groq LLM
def get_response(messages):
    result = run_agent(messages)
    
    return result["final_answer"]

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hello! I'm your friendly AI assistant. How can I help you today?")
    ]

# Display chat messages from history
for message in st.session_state.messages:
    # Use st.chat_message to display messages with roles
    with st.chat_message(message.type):
        st.markdown(message.content)

# Get user input from the chat input box
if prompt := st.chat_input("What's on your mind?"):
    # Add user message to session state and display it
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("human"):
        st.markdown(prompt)

    # Get the AI response and display it
    with st.spinner("Thinking..."):
        ai_response_content = get_response(st.session_state.messages)
        ai_response = AIMessage(content=ai_response_content)
        st.session_state.messages.append(ai_response)
        with st.chat_message("ai"):
            st.markdown(ai_response_content)