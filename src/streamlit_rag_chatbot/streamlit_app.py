import streamlit as st
import logging  # Import logging
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import uuid
from main import chatbot
import streamlit_nested_layout

# Set up logging
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
  
def query_meeting_summary(datestamp):
    
    bot = st.session_state.get("bot")
    
    # Query summaries from the bot
    result = bot.query_summaries(datestamp)

    # Extract details safely from metadata
    date = result.get("date", "Unknown date")
    speakers_list = result.get("speakers", [])
    summary = result.get("summary", "No summary available")

    # Construct speaker string
    if isinstance(speakers_list, list) and len(speakers_list) > 1:
        speakers = ' and '.join([', '.join(speakers_list[:-1]), speakers_list[-1]])
    else:
        speakers = speakers_list[0] if speakers_list else "No speakers listed"

    # Construct and return the output
    return (
        f"**Meeting date:** {date}.\n\n"
        f"**Speaker list:** {speakers}.\n\n"
        f"**Meeting summary:** {summary}"
    )



st.title(f"🤖 EER Transcript Explorer Bot")
st.write("""
    This is a chatbot with access to meeting transcripts from the EER project (May 2021 - December 2024) and relevant project documents. The first part of the chatbot's answer to your question refers to the transcripts and other source data. The second part describes connections between your question and questions other people have asked about the data. Perhaps you'll learn that someone else is curious about similar things. Please note that all interactions are stored in a database and will be visible to other users.
""")

with st.expander("Fetch a meeting summary", expanded=False):
    #datestamp = st.date_input("Meeting summary date", datetime.date(2024, 1, 30))
    
    datestamp = st.selectbox(
        "Select a date to fetch the meeting summary",
        ("2024-12-10","2024-11-24", "2024-11-19", "2024-11-12", "2024-10-29", "2024-10-24",
         "2024-10-15", "2024-10-01", "2024-09-24", "2024-09-17", "2024-09-10",
        "2024-08-22", "2024-08-15", "2024-08-06", "2024-06-25", "2024-06-11",
        "2024-05-28", "2024-05-21", "2024-05-07", "2024-04-30", "2024-04-23",
        "2024-04-09", "2024-03-26", "2024-03-19", "2024-03-12", "2024-03-05",
        "2024-02-27", "2024-01-30", "2024-01-25", "2024-01-24", "2024-01-18",
        "2024-01-11", "2023-12-07", "2023-11-30", "2023-11-23", "2023-10-19",
        "2023-10-05", "2023-09-28", "2023-09-21", "2023-08-31", "2023-08-24",
        "2023-08-17", "2023-08-10", "2023-06-29", "2023-06-22", "2023-06-15",
        "2023-06-08", "2023-06-01", "2023-05-25", "2023-05-04", "2023-04-27",
        "2023-04-20", "2023-04-13", "2023-03-23", "2023-03-16", "2023-02-02",
        "2023-01-19", "2023-01-05", "2022-12-16", "2022-12-08", "2022-12-01",
        "2022-11-24", "2022-11-17", "2022-11-10", "2022-10-20", "2022-10-13",
        "2022-09-29", "2022-09-08", "2022-09-01", "2022-08-25", "2022-06-30",
        "2022-05-05", "2022-04-21", "2022-04-11", "2022-03-24", "2022-03-10",
        "2022-02-24", "2022-02-10", "2022-02-02", "2022-01-20", "2022-01-13",
        "2022-01-04", "2021-12-09", "2021-12-02", "2021-11-25", "2021-11-18",
        "2021-11-11", "2021-11-04", "2021-09-30", "2021-09-23", "2021-09-03",
        "2021-08-13", "2021-06-25", "2021-06-18", "2021-06-11", "2021-06-04",
        "2021-05-28")
        )

    # Fetch the summary
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
                                with st.expander("Past conversations with this chatbot related to this topic", expanded=False):
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
                        with st.expander("Past conversations with this chatbot related to this topic", expanded=False):
                            for idx, doc in enumerate(past_chat_context, 1):
                                with st.expander(f"User question: _\"{doc.metadata.get('user_question')}\"_", expanded=False):
                                    st.markdown(f"**User name:** {doc.metadata.get('user_name', 'Unknown user name')}")
                                    st.markdown(f"**AI Response:** {doc.metadata.get('ai_output')}")
                                    st.markdown(f"**Date:** {doc.metadata.get('date', 'Unknown date')}")
    except Exception as e:
        st.error(f"Error generating response: {e}")
        logger.error(f"Error during input handling: {e}")