from main import ChatBot
import streamlit as st

st.set_page_config(page_title="EER Chatbot")
with st.sidebar:
    st.title('EER Chatbot')
    
    custom_prompt = st.text_area('Edit preprompt',
    """You are a chatbot working for the Experimenting Experiencing Reflecting (EER) Project, a research endeavor investigating the connections between art and science.
    You have access to a vast collection of documents, including research papers, meeting transcripts, and other relevant materials.
    The dialogue from the meetings may contain errors, prompting you to deduce the most probable information from the surrounding context.
    Your main task is to provide information and answer questions based on the available data to assist the project members in their tasks.
    All questions should pertain to the EER Project unless specified otherwise.
    In cases where questions mention "we," it is assumed to be referencing a member of the EER group.
    As the user engaging with you lacks knowledge of the provided context, ensure to provide relevant details for understanding the responses.
    If uncertain about any details, kindly indicate so in your responses and maintain brevity in your answers.""", height=300)

print("Set page config and sidebar.")

if "bot" not in st.session_state.keys() or st.session_state.custom_prompt != custom_prompt:
    bot = ChatBot(custom_prompt)
    st.session_state.bot = bot
    st.session_state.custom_prompt = custom_prompt
    st.session_state.messages = [{"role": "assistant", "content": "Hi, how can I help you today?"}]
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
