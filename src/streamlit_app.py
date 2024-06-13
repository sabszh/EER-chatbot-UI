from main import ChatBot
import streamlit as st

repositories = {
    "Mistral-7B-Instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
    "Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Mistral-7B-Instruct-v0.1": "mistralai/Mistral-7B-Instruct-v0.1"
}

st.set_page_config(page_title="EER Chatbot")

with st.sidebar:
    st.title('EER Chatbot')
    st.write("""Be aware, if you make any changes here that the chatbot will reload and your chat will be gone.
             If you get an error, try a different model. If that does not work, it might be overloaded or down - so try again later.""")
    selected_repo = st.selectbox("Select the Model Repository", list(repositories.keys()))
    temperature = st.slider("Select the Temperature (0-2)", min_value=0.1, max_value=2.0, value=1.0, step=0.01)
    custom_prompt = st.text_area('Edit Preprompt',
    """You are a chatbot working for the Experimenting Experiencing Reflecting (EER) Project, a research endeavor investigating the connections between art and science.
    You have access to a collection of documents, including descriptions of research activities, meeting transcripts, and other relevant materials.
    Your main task is to help the user explore and reflect on the EER project.
    All questions should pertain to the EER Project unless specified otherwise.
    When possible, please cite source documents in your answer (calling the documents the transcript date, e.g. "2021-05-28", or website page after the "/").""", height=250)

# Initialize chat history if not present
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi, how can I help you today?"}]
if "first_question" not in st.session_state:
    st.session_state.first_question = ""

# Initialize ChatBot instance if needed
def initialize_bot():
    repo_id = repositories[selected_repo]
    bot = ChatBot(custom_template=custom_prompt, repo_id=repo_id, temperature=temperature)
    st.session_state.bot = bot
    st.session_state.custom_prompt = custom_prompt
    st.session_state.selected_repo = selected_repo
    st.session_state.temperature = temperature
    st.session_state.repo_id = repo_id

if "bot" not in st.session_state or st.session_state.custom_prompt != custom_prompt or st.session_state.selected_repo != selected_repo or st.session_state.temperature != temperature:
    initialize_bot()
else:
    bot = st.session_state.bot

# Function for generating LLM response
def generate_response(input):
    result = bot.rag_chain.invoke(input)
    return result

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle user input
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

    # Store the first question if not already set
    if not st.session_state.first_question:
        st.session_state.first_question = input

    # Generate response if needed
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Generating an answer.."):
                response = generate_response(input) 
                st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
