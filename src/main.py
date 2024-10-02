from dotenv import load_dotenv
from datetime import datetime, timezone
import os

from data_chunking import datachunk
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint as HuggingFaceHub
from langchain_community.vectorstores.pinecone import Pinecone
from pinecone import Pinecone as pc

class chatbot:
    def __init__(self, temperature=0.8, prompt_sourcedata=None, prompt_conv=None, user_name=None, session_id=None):
        load_dotenv()
        self.embeddings = HuggingFaceEmbeddings()
        self.index_name = "eer-transcripts-pdfs"
        self.pinecone_instance = pc(api_key=os.getenv('PINECONE_API_KEY'), embeddings=self.embeddings)
        
        # Self-assign parameters
        self.user_name = user_name
        self.session_id = session_id
        
        self.temperature = temperature
        
        # Instantiate the LLM
        self.llm = HuggingFaceHub(
            repo_id=os.getenv('repo_id'),
            temperature=temperature,
            top_p=0.8,
            top_k=50,
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )

    # Renamed method
    def default_prompt_sourcedata(self, chat_history, original_data, user_input, user_name):
        return f"""You are a chatbot assistant, working for the Experimenting Experiencing Reflecting (EER) Project, a research endeavor investigating the connections between art and science.
    You have access to a collection of documents, including descriptions of research activities, meeting transcripts, and other relevant materials.Your main task is to help the user explore and reflect on the EER project.     When possible, please cite source documents in your answer (calling the documents the transcript date, e.g. "2021-05-28", or website page after the "/". 
    You are now assisting the user "{user_name}" with their query: "{user_input}". Below is the relevant data from the EER group that was retrieved from the database for this query: "{original_data}". Based on this data, provide a concise answer to the user’s question in 1-3 sentences. The previous chat history for this session so far is: {chat_history} Your response:"""

    def default_prompt_conv(self, chat_history, user_input, llm_response, past_chat, user_name):
        return f"""You are an assistant observing conversations between the user and another LLM regarding meeting transcripts and content from the Experimenting Experiencing Reflecting (EER) Project, a research group investigating the connections between art and science.
    All interactions between people and the LLM are recorded and stored in your database. When people ask questions about the data, you get the question and the answer from the LLM. You use that data to search your database of past conversations for conversations that might be related. You will create a summary of those past conversations no longer than 4 sentences. Your summary should mention the name of the person involved in the past conversations, so that if the user wants to, they can follow up with them.
    Here is the last question asked by the user in this session: "{user_name}" asked: {user_input}
    Here is what the LLM you are watching responded with: “{llm_response}”
    Here is relevant data from past conversations that is relevant: {past_chat}
    Here is the chat history for this session, so that your response can be aware of the context: {chat_history}
    Your response: """

   # Method to retrieve documents from Pinecone index while excluding a specific session_id
    def retrieve_docs(self, input, index, excluded_session_id=None, k=5):
        # Retrieve past conversation data
        docsearch = Pinecone.from_existing_index(index, self.embeddings)
        
        if index == "eer-interaction-data":
            # Add metadata filter to exclude the given session_id
            search_kwargs = {
                "k": k,
                "filter": {
                    "session_id": {"$ne": excluded_session_id}
                }
            }
        else:
            search_kwargs = {
                "k": k
            }
            
        retriever = docsearch.as_retriever(search_kwargs=search_kwargs)
        docs = retriever.invoke(input)
        
        return docs

    # Method to generate response from LLM
    def get_llm_response(self, prompt):
        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            error = f"Error invoking LLM: {e}"
            return error
    
    # Method to format the context
    def format_context(self, documents, chat=False):
        context = ""
        
        for idx, doc in enumerate(documents, start=1):
            metadata = doc.metadata
            if chat:
                context += (
                    f'User {idx}: {metadata.get("user_name", "Unknown User")}\n'
                    f'Chat session {idx}: {metadata.get("session_id", "Unknown Session ID")}\n'
                    f'User Question: "{metadata.get("user_question", "Unknown Question")}"\n'
                    f'AI Response: "{metadata.get("ai_output", "Unknown Response")}"\n'
                    f"Date: {metadata.get('date', 'Unknown Date')}\n\n"
                )
            else:
                if metadata.get("page") is not None:
                    context += (
                        f"Document type: PDF {idx}\n"
                        f"Page content: {doc.page_content}\n"
                        f"Source: {metadata['source']}\n"
                        f"Page: {metadata['page']}\n"
                    )
                else:
                    context += (
                        f"Document type: Meeting Transcript Exerpt {idx}\n"
                        f"Person {idx}: {metadata.get('speaker_name', 'Unknown Speaker')}\n"
                        f"Date: {metadata.get('date_time', 'Unknown Date')}\n"
                        f"Content: {doc.page_content}\n\n"
                    )

        return context

    # Method to upsert data to Pinecone index
    def upsert_vectorstore(self, user_input, ai_output, user_name, session_id):
        # Pinecone index for chat data
        pinecone_instance_chat = pc(api_key=os.getenv('PINECONE_API_KEY'), embeddings=self.embeddings)
        index_name = "eer-interaction-data"
        environment = "gcp-starter"
        
        index = pinecone_instance_chat.Index(index_name, environment=environment)
                
        embedding = self.embeddings.embed_documents([user_input + ai_output])[0]
        
        # Upsert the summary embedding to the Pinecone index with metadata, including timestamp
        index.upsert(vectors=[
            {
                'id': session_id,
                'values': embedding,
                'metadata': {
                    "user_question": user_input,
                    "ai_output": ai_output,
                    "user_name": user_name,
                    "session_id": session_id,
                    "date": datetime.now(timezone.utc).isoformat(),
                    "text": f"User input: {user_input}, AI output: {ai_output}"
                }
            }
        ])

    def pipeline(self, user_input, user_name, session_id, chat_history=None):
        # Step 0: Add chat history to the context
        if chat_history:
            chat_history = chat_history + "\n\n"
        else:
            chat_history = ""
        
        # Step 1: Retrieve source data
        source_data = self.retrieve_docs(user_input, "eer-transcripts-pdfs")
        formatted_source_data = self.format_context(source_data)

        # Step 2: Generate LLM response from source data
        sourcedata_response = self.get_llm_response(self.default_prompt_sourcedata(chat_history=chat_history, original_data = formatted_source_data, user_input = user_input, user_name=user_name))

        # Step 3: Retrieve past chat context
        past_chat_context = self.retrieve_docs(sourcedata_response, "eer-interaction-data", session_id)
        formatted_chat_context = self.format_context(past_chat_context, chat=True)
        
        # Step 5: Generate LLM response for conversation context, now considering combined chat history
        conversation_response = self.get_llm_response(self.default_prompt_conv(chat_history=chat_history, user_input=user_input, llm_response=sourcedata_response, past_chat=formatted_chat_context, user_name=user_name))

        # Step 6: Combine the responses
        ai_output = f"{sourcedata_response}\n\n{conversation_response}"
        
        print("AI Output: ", ai_output)
        print("sourcedaa_response: ", sourcedata_response)
        print("conversation_response: ", conversation_response)

        # Upsert to vector store
        self.upsert_vectorstore(user_input, ai_output, user_name, session_id)

        # Return a dictionary containing all relevant information
        return {
            "ai_output": ai_output,
            "source_data": source_data,
            "past_chat_context": past_chat_context
        }