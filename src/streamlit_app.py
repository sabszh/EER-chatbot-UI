import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import uuid
from main import chatbot
import streamlit_nested_layout

st.set_page_config(page_title="EER Transcript Explorer Bot", layout="wide")

# Initialize chat history if not present
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi, how can I help you today?"}]
if "first_question" not in st.session_state:
    st.session_state.first_question = ""

# Initialize ChatBot instance if needed
def initialize_bot():
    bot = chatbot()

# Initialize session state variables
if "chat_data" not in st.session_state:
    st.session_state.chat_data = []

if "user_name" not in st.session_state:
    st.session_state.user_name = None

if "session_id" not in st.session_state:
    # Generate a random session ID using uuid
    st.session_state.session_id = str(uuid.uuid4())

# Initialize history if it doesn't exist
if "history" not in st.session_state:
    st.session_state.history = StreamlitChatMessageHistory()

# Function to capture and set the user's name
@st.dialog("Please enter your name:", width="small")
def ask_name():
    user_name = st.text_input("")
    if st.button("Submit"):
        if user_name:
            st.session_state.user_name = user_name
            st.rerun()

if st.session_state.user_name is None:
    ask_name()
    
# Initialize the chatbot instance if required
def initialize_bot():
    
    try:
        if ("bot" not in st.session_state or st.session_state.bot is None):
            st.session_state.bot = chatbot()
            
    except Exception as e:
        st.error(f"Error initializing bot: {e}")
        print(f"Error initializing bot: {e}")

initialize_bot()

# Function to generate a response from the chatbot
def generate_response(input_text):
    bot = st.session_state.get("bot")
    
    # Include chat history in the response generation
    chat_history = "\n".join([
        f"User: {msg.get('input_text', '')}\nAI: {msg.get('ai_output', '')}" 
        for msg in st.session_state.chat_data 
        if msg.get('type') == 'ai' or msg.get('type') == 'user'
        ])
    
    result = bot.pipeline(user_input=input_text, user_name=st.session_state.user_name, session_id=st.session_state.session_id, chat_history=chat_history)
    
    return result

# Main chat interface
st.title(f"🤖 EER Transcript Explorer Bot")
st.write("""
        This is a chatbot with access to meeting transcripts from the EER project (May 2021 - January 2024) and relevant project documents. The first part of the chatbot's answer to your question refers to the transcripts and other source data. The second part describes connections between your question and questions other people have asked about the data. Perhaps you'll learn that someone else is curious about similar things. Please note that all interactions are stored in a database and will be visible to other users.
 """)

chat_container = st.container()

with chat_container.chat_message("ai"):
    st.write(f"Hi {st.session_state.user_name}, what would you like to ask me about the EER project?")

# Display all previous chat messages
with chat_container:
    for entry in st.session_state.chat_data:
        entry_type = entry.get("type")
        ai_output = entry.get("ai_output")
        user_input = entry.get("input_text", "")
        source_data = entry.get("source_data", [])
        past_chat_context = entry.get("past_chat_context", [])

        if entry_type == "user":
            with st.chat_message("user"):
                st.write(user_input)  # Display user message
        elif entry_type == "ai":
            with st.chat_message("ai"):
                st.write(ai_output)  # Display AI response

                with st.expander("Referenced data", expanded=False):
                    with st.expander("Transcripts and documents", expanded=False):
                        for idx, doc in enumerate(source_data, start=1):
                            metadata = doc.metadata
                            # Check if it's a PDF or a meeting transcript
                            if metadata.get("page") is not None:
                                with st.expander(f"PDF Document {idx} - Page {metadata['page']}"):
                                    st.markdown(f"**Source:** {metadata.get('source', 'Unknown source')}")
                                    st.markdown(f"**Content:** {doc.page_content}")
                                    st.markdown(f"**Page:** {metadata.get('page', 'Unknown page')}")   
                            else:
                                with st.expander(f"Meeting Transcript {idx} - {metadata.get('speaker_name', 'Unknown Speaker')}"):
                                    st.markdown(f"**Content:** {doc.page_content}")
                                    st.markdown(f"**Speaker Name:** {metadata.get('speaker_name', 'Unknown Speaker')}")
                                    st.markdown(f"**Date:** {metadata.get('date_time', 'Unknown date')}")
                                    
                    # Expander for Previous Chat (Past Memory)
                    if past_chat_context:
                        with st.expander("Data from previous conversations with this LLM", expanded=False):
                            for idx, doc in enumerate(past_chat_context, 1):
                                with st.expander(f"User question: _\"{doc.metadata.get('user_question')}\"_", expanded=False):
                                    st.markdown(f"**User name:** {doc.metadata.get('user_name', 'Unknown user name')}")
                                    st.markdown(f"**AI Response:** {doc.metadata.get('ai_output')}")
                                    st.markdown(f"**Date:** {doc.metadata.get('date', 'Unknown date')}")
# Handle user input
input_text = st.chat_input("Type your message here...")

if input_text:
    try:
        # Append user message to chat_data
        st.session_state.chat_data.append({
            "type": "user",
            "user_name": st.session_state.user_name,
            "input_text": input_text,
            "session_id": st.session_state.session_id,
            "retrieved_docs": []
        })
        
        # Display user message immediately
        with chat_container.chat_message("user"):
            st.write(input_text)

        # Generate AI response
        with st.spinner("Thinking..."):
            result = generate_response(input_text)
            ai_output = result.get("ai_output", "No answer generated")
            source_data = result.get("source_data", [])
            past_chat_context = result.get("past_chat_context", [])

            # Append AI response to chat_data
            st.session_state.chat_data.append({
                "type": "ai",
                "ai_output": ai_output,
                "source_data": source_data,
                "past_chat_context": past_chat_context
            })

            # Display AI message
            with chat_container.chat_message("ai"):
                st.write(ai_output)
                
                 # Expander for Memories (Referenced Data)
                with st.expander("Referenced data", expanded=False):
                    with st.expander("Transcripts and documents", expanded=False):
                        for idx, doc in enumerate(source_data, start=1):
                            metadata = doc.metadata
                            # Check if it's a PDF or a meeting transcript
                            if metadata.get("page") is not None:
                                with st.expander(f"PDF Document {idx} - Page {metadata['page']}"):
                                    st.markdown(f"**Source:** {metadata.get('source', 'Unknown source')}")
                                    st.markdown(f"**Content:** {doc.page_content}")
                                    st.markdown(f"**Page:** {metadata.get('page', 'Unknown page')}")   
                            else:
                                with st.expander(f"Meeting Transcript {idx} - {metadata.get('speaker_name', 'Unknown Speaker')}"):
                                    st.markdown(f"**Content:** {doc.page_content}")
                                    st.markdown(f"**Speaker Name:** {metadata.get('speaker_name', 'Unknown Speaker')}")
                                    st.markdown(f"**Date:** {metadata.get('date_time', 'Unknown date')}")
                                    
                    # Expander for Previous Chat (Past Memory)
                    if past_chat_context:
                        with st.expander("Data from previous conversations with this LLM", expanded=False):
                            for idx, doc in enumerate(past_chat_context, 1):
                                with st.expander(f"User question: _\"{doc.metadata.get('user_question')}\"_", expanded=False):
                                    st.markdown(f"**User name:** {doc.metadata.get('user_name', 'Unknown user name')}")
                                    st.markdown(f"**AI Response:** {doc.metadata.get('ai_output')}")
                                    st.markdown(f"**Date:** {doc.metadata.get('date', 'Unknown date')}")
                                   
    except Exception as e:
            st.error(f"Error generating response: {e}")
