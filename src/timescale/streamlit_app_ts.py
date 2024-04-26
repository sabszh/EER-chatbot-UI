import streamlit as st
from main_ts import ChatBot, SimilarityRetriever
from datetime import datetime

repositories = {
    "Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Mistral-7B-Instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
    "Mistral-7B-Instruct-v0.1": "mistralai/Mistral-7B-Instruct-v0.1"
}

# Set up the page configuration
st.set_page_config(page_title="EER Chatbot")
with st.sidebar:
    st.title('EER Chatbot')
    st.write("Be aware, if you make any changes here that the chatbot will reload and your chat will be gone.")
    selected_repo = st.selectbox("Select the Model Repository", list(repositories.keys()))

    temperature = st.slider("Select the Temperature (0-2)", min_value=0.1, max_value=2.0, value=1.0, step=0.01)
    
    custom_prompt = st.text_area('Edit System Prompt',
    """You are a chatbot working for the Experimenting Experiencing Reflecting (EER) Project, a research endeavor investigating the connections between art and science.
    You have access to a vast collection of documents, including research papers, meeting transcripts, and other relevant materials.
    The dialogue from the meetings may contain errors, prompting you to deduce the most probable information from the surrounding context.
    Your main task is to provide information and answer questions based on the available data to assist the project members in their tasks.
    All questions should pertain to the EER Project unless specified otherwise.
    In cases where questions mention "we," it is assumed to be referencing a member of the EER group.
    As the user engaging with you lacks knowledge of the provided context, ensure to provide relevant details for understanding the responses.
    If uncertain about any details, kindly indicate so in your responses and maintain brevity in your answers.""", height=250)
    
    st.write("You can adjust time and amounts of documents to include in similarity search in the same session")
    
    k_value = st.number_input("Number of documents to include in similarity search", min_value=5, max_value=20, value=5)
    start_date = st.date_input("Start Date", datetime(2021, 5, 28), key="start_date") 
    end_date = st.date_input("End Date", datetime.now(), key="end_date")
    

# Initialize ChatBot based on selected repository and temperature, and k_value
if "bot" not in st.session_state.keys() or st.session_state.custom_prompt != custom_prompt or st.session_state.selected_repo != selected_repo or st.session_state.temperature != temperature:
    repo_id = repositories[selected_repo]
    bot = ChatBot(custom_template=custom_prompt, repo_id=repo_id, temperature=temperature)
    st.session_state.bot = bot
    st.session_state.custom_prompt = custom_prompt
    st.session_state.selected_repo = selected_repo
    st.session_state.temperature = temperature
    st.session_state.k_value = k_value
    st.session_state.messages = [{"role": "assistant", "content": "Hi, how can I help you today?"}]
    st.session_state.repo_id = repo_id
else:
    bot = st.session_state.bot

# Function for generating LLM response
def generate_response(input, start_date, end_date, session_messages, k_value):
    # Convert start_date and end_date to the required format (string)
    start_date_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
    end_date_str = end_date.strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare input with question, dates, and session messages
    input_with_dates = {"question": input, "start_date": start_date_str, "end_date": end_date_str, "session_messages": session_messages}
    
    # Generate response
    response = bot.rag_chain.invoke(input_with_dates,config={"k": bot.k_value})
    context = SimilarityRetriever().invoke(input_with_dates, config={"k": k_value})

    return response, context

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hi, how can I help you today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a new response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Generating an answer.."):
            response, context = generate_response(input, start_date, end_date, st.session_state.messages, k_value)
        st.write(response)
    # Add the context, user input, and assistant's response to the history
    message = {"role": "assistant", "content": response, "context": context, "input": input}
    st.session_state.messages.append(message)