import streamlit as st
import logging  # Import logging
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import uuid
from main import chatbot
import streamlit_nested_layout
import datetime

# Set up logging to only log user input, AI responses, and errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot_log.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="EER Transcript Explorer Bot", layout="wide")

# Initialize chat history if not present
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi, how can I help you today?"}]
if "first_question" not in st.session_state:
    st.session_state.first_question = ""

# Functiion to initialize the chatbot
def initialize_bot():
    try:
        if "bot" not in st.session_state or st.session_state.bot is None:
            st.session_state.bot = chatbot()
    except Exception as e:
        st.error(f"Error initializing bot: {e}")
        logger.error(f"Error initializing bot: {e}")

initialize_bot()

# Setting up the session state variables
if "chat_data" not in st.session_state:
    st.session_state.chat_data = []

if "user_name" not in st.session_state:
    st.session_state.user_name = None

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "history" not in st.session_state:
    st.session_state.history = StreamlitChatMessageHistory()

# Function to ask the user for their name
@st.dialog("Please enter your name:", width="small")
def ask_name():
    user_name = st.text_input(label="Name", label_visibility="collapsed", placeholder="Your name")
    if st.button("Submit"):
        if user_name:
            st.session_state.user_name = user_name
            logger.info(f"User name set to: {user_name}")
            st.rerun()

if st.session_state.user_name is None:
    ask_name()

# Function to generate a response from the AI
def generate_response(input_text):
    bot = st.session_state.get("bot")
    chat_history = "\n".join([
        f"User: {msg.get('input_text', '')}\nAI: {msg.get('ai_output', '')}"
        for msg in st.session_state.chat_data
        if msg.get('type') == 'ai' or msg.get('type') == 'user'
    ])
    try:
        result = bot.pipeline(user_input=input_text, user_name=st.session_state.user_name, session_id=st.session_state.session_id, chat_history=chat_history)
        logger.info(f"AI Response: {result.get('ai_output', 'No answer generated')}")
        return result
    except Exception as e:
        st.error(f"Error generating response: {e}")
        logger.error(f"Error generating response: {e}")
        return {}
    
# Function to query the meeting summary for a given date
def query_meeting_summary(datestamp):
    bot = st.session_state.get("bot")
    
    # Convert datestamp to string
    datestamp_str = datestamp.strftime("%Y-%m-%d")
    
    result = bot.query_summaries(datestamp_str)
    
    date = result["matches"][0]["metadata"]["date"]
    speakers = ' and '.join([', '.join(result["matches"][0]["metadata"]["speakers"][:-1]), result["matches"][0]["metadata"]["speakers"][-1]]) if len(result["matches"][0]["metadata"]["speakers"]) > 1 else result["matches"][0]["metadata"]["speakers"][0]
    summary = result["matches"][0]["metadata"]["summary"]
    
    return f"**Meeting date:** {date}. \n\n **Speaker list:** {speakers}. \n\n  **Meeting summary:**{summary}"

st.title(f"🤖 EER Transcript Explorer Bot")
st.write("""
    This is a chatbot with access to meeting transcripts from the EER project (May 2021 - January 2024) and relevant project documents. The first part of the chatbot's answer to your question refers to the transcripts and other source data. The second part describes connections between your question and questions other people have asked about the data. Perhaps you'll learn that someone else is curious about similar things. Please note that all interactions are stored in a database and will be visible to other users.
""")

with st.expander("Fetch a meeting summary", expanded=False):
    datestamp = st.date_input("Meeting summary date", datetime.date(2024, 1, 30))
    summary = query_meeting_summary(datestamp)
    st.info(summary, icon="📄")

chat_container = st.container()

with chat_container.chat_message("ai"):
    st.write(f"Hi {st.session_state.user_name}, what would you like to ask me about the EER project?")

with chat_container:
    for entry in st.session_state.chat_data:
        entry_type = entry.get("type")
        ai_output = entry.get("ai_output")
        user_input = entry.get("input_text", "")
        source_data = entry.get("source_data", [])
        past_chat_context = entry.get("past_chat_context", [])

        if entry_type == "user":
            with st.chat_message("user"):
                st.write(user_input)
        elif entry_type == "ai":
            with st.chat_message("ai"):
                st.write(ai_output)
                with st.expander("Referenced data", expanded=False):
                    with st.expander("Transcripts and documents", expanded=False):
                        for idx, doc in enumerate(source_data, start=1):
                            metadata = doc.metadata
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
                            if past_chat_context:
                                with st.expander("Data from previous conversations with this LLM", expanded=False):
                                    for idx, doc in enumerate(past_chat_context, 1):
                                        with st.expander(f"User question: _\"{doc.metadata.get('user_question')}\"_", expanded=False):
                                            st.markdown(f"**User name:** {doc.metadata.get('user_name', 'Unknown user name')}")
                                            st.markdown(f"**AI Response:** {doc.metadata.get('ai_output')}")
                                            st.markdown(f"**Date:** {doc.metadata.get('date', 'Unknown date')}")

input_text = st.chat_input("Type your message here...")

if input_text:
    try:
        logger.info(f"User Question: {input_text}")
        st.session_state.chat_data.append({
            "type": "user",
            "user_name": st.session_state.user_name,
            "input_text": input_text,
            "session_id": st.session_state.session_id,
            "retrieved_docs": []
        })
        with chat_container.chat_message("user"):
            st.write(input_text)

        with st.spinner("Thinking..."):
            result = generate_response(input_text)
            ai_output = result.get("ai_output", "No answer generated")
            source_data = result.get("source_data", [])
            past_chat_context = result.get("past_chat_context", [])

            st.session_state.chat_data.append({
                "type": "ai",
                "ai_output": ai_output,
                "source_data": source_data,
                "past_chat_context": past_chat_context
            })

            with chat_container.chat_message("ai"):
                st.write(ai_output)
                with st.expander("Referenced data", expanded=False):
                    with st.expander("Transcripts and documents", expanded=False):
                        for idx, doc in enumerate(source_data, start=1):
                            metadata = doc.metadata
                            if metadata.get("page") is not None:
                                with st.expander(f"PDF Document {idx}: Page {metadata['page']}"):
                                    st.markdown(f"**Source:** {metadata.get('source', 'Unknown source')}")
                                    st.markdown(f"**Content:** {doc.page_content}")
                                    st.markdown(f"**Page:** {metadata.get('page', 'Unknown page')}")
                            else:
                                with st.expander(f"EER Meeting: {metadata.get('date_time', 'Unknown date')}, {metadata.get('speaker_name', 'Unknown Speaker')}"):
                                    st.markdown(f"**Content:** {doc.page_content}")
                                    st.markdown(f"**Speaker Name:** {metadata.get('speaker_name', 'Unknown Speaker')}")
                                    st.markdown(f"**Date:** {metadata.get('date_time', 'Unknown date')}")
                    if past_chat_context:
                        with st.expander("Data from previous conversations with this LLM", expanded=False):
                            for idx, doc in enumerate(past_chat_context, 1):
                                with st.expander(f"User question: _\"{doc.metadata.get('user_question')}\"_", expanded=False):
                                    st.markdown(f"**User name:** {doc.metadata.get('user_name', 'Unknown user name')}")
                                    st.markdown(f"**AI Response:** {doc.metadata.get('ai_output')}")
                                    st.markdown(f"**Date:** {doc.metadata.get('date', 'Unknown date')}")
    except Exception as e:
        st.error(f"Error generating response: {e}")
        logger.error(f"Error during input handling: {e}")