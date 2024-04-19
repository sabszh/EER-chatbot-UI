from main import ChatBot
import streamlit as st

repositories = {
    "Mistral-7B-Instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
    "Mistral-7B-Instruct-v0.1": "mistralai/Mistral-7B-Instruct-v0.1",
    "Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1"
}

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

print("Set page config and sidebar.")

# Initialize ChatBot based on selected repository and temperature
if "bot" not in st.session_state.keys() or st.session_state.custom_prompt != custom_prompt or st.session_state.selected_repo != selected_repo or st.session_state.temperature != temperature:
    repo_id = repositories[selected_repo]
    bot = ChatBot(custom_template=custom_prompt, repo_id=repo_id, temperature=temperature)
    st.session_state.bot = bot
    st.session_state.custom_prompt = custom_prompt
    st.session_state.selected_repo = selected_repo
    st.session_state.temperature = temperature
    st.session_state.messages = [{"role": "assistant", "content": "Hi, how can I help you today?"}]
    st.session_state.repo_id = repo_id
    print("Initialized session_state.")
else:
    bot = st.session_state.bot

print("Initialized ChatBot.")

# Function for generating LLM response
def generate_response(input):
    result = bot.rag_chain.invoke(input)
    return result

print("Defined generate_response function.")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hi, how can I help you today?"}]
    print("Initialized session_state messages.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        print(f"Displayed message: {message['content']}")

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)
        print(f"User input: {input}")

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Generating an answer.."):
            response = generate_response(input) 
            st.write(response)
            print(f"Generated response: {response}")
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
    print(f"Appended assistant response to session_state messages.")
