import os
from datetime import datetime, timezone
import logging
from tenacity import retry, stop_after_attempt, wait_fixed
from dotenv import load_dotenv
from data_chunking import datachunk
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint as HuggingFaceHub
from langchain_community.vectorstores.pinecone import Pinecone
from pinecone import Pinecone as pc
import pinecone

logger = logging.getLogger(__name__)

class chatbot:
    """
    A chatbot class that interacts with the Experimenting Experiencing Reflecting (EER) project data, including document retrieval, LLM integration, and storing chat history.

    Attributes:
        temperature (float): The temperature parameter for the language model.
        user_name (str): The name of the user interacting with the chatbot.
        session_id (str): A unique identifier for the user's session.
        embeddings (HuggingFaceEmbeddings): Embedding generator for document vectors.
        llm (HuggingFaceHub): HuggingFace language model endpoint.
    """

    def __init__(self, temperature=0.8, prompt_sourcedata=None, prompt_conv=None, user_name=None, session_id=None):
        """
        Initializes the chatbot instance with parameters and sets up embeddings and LLM.

        Args:
            temperature (float, optional): Temperature for controlling randomness in LLM outputs. Defaults to 0.8.
            prompt_sourcedata (str, optional): Prompt template for source data queries.
            prompt_conv (str, optional): Prompt template for conversational context.
            user_name (str, optional): Name of the user.
            session_id (str, optional): Unique session identifier.
        """
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
        self.prompt_sourcedata = prompt_sourcedata
        self.prompt_conv = prompt_conv

    def default_prompt_sourcedata(self, chat_history, original_data, user_input, user_name):
        """
        Generates the default prompt for sourced data queries.

        Args:
            chat_history (str): The chat history so far.
            original_data (str): The relevant data retrieved from the database.
            user_input (str): The user's query.
            user_name (str): The name of the user.

        Returns:
            str: The formatted prompt for sourced data queries.
        """
        return f"""You are a chatbot assistant, working for the Experimenting Experiencing Reflecting (EER) Project, a research endeavor investigating the connections between art and science.
    You have access to a collection of documents, including descriptions of research activities, meeting transcripts, and other relevant materials. Your main task is to help the user explore and reflect on the EER project. When possible, please cite source documents in your answer (calling the documents the transcript date, e.g. "2021-05-28", or website page after the "/"). 
    You are now assisting the user "{user_name}" with their query: "{user_input}". Below is the relevant data from the EER group that was retrieved from the database for this query: "{original_data}". Based on this data, provide a concise answer to the user’s question in maximum 2 paragraphs. The previous chat history for this session so far is: {chat_history} Your response:"""

    def default_prompt_conv(self, chat_history, user_input, llm_response, past_chat, user_name):
        """
        Generates the default prompt for conversation context queries.

        Args:
            chat_history (str): The chat history so far.
            user_input (str): The user's query.
            llm_response (str): The response from the LLM.
            past_chat (str): Relevant data from past conversations.
            user_name (str): The name of the user.

        Returns:
            str: The formatted prompt for conversation context queries.
        """
        return f"""You are an assistant observing conversations between the user and another LLM regarding meeting transcripts and content from the Experimenting Experiencing Reflecting (EER) Project, a research group investigating the connections between art and science.
    All interactions between people and the LLM are recorded and stored in your database. When people ask questions about the data, you get the question and the answer from the LLM. You use that data to search your database of past conversations for conversations that might be related. You will create a summary of those past conversations no longer than 4 sentences. Your summary should mention the name of the person involved in the past conversations, so that if the user wants to, they can follow up with them.
    Here is the last question asked by the user in this session: "{user_name}" asked: {user_input}
    Here is what the LLM you are watching responded with: “{llm_response}”
    Here is relevant data from past conversations that is relevant: {past_chat}
    Here is the chat history for this session, so that your response can be aware of the context: {chat_history}
    Your response: """

    def retrieve_docs(self, input, index, excluded_session_id=None, k=5):
        """
        Retrieves documents from a Pinecone index, optionally excluding a specific session ID.

        Args:
            input (str): The input query for retrieving documents.
            index (str): The name of the Pinecone index to retrieve documents from.
            excluded_session_id (str, optional): Session ID to exclude from retrieval (for avoiding duplicate data).
            k (int, optional): The number of documents to retrieve. Defaults to 5.

        Returns:
            list: A list of retrieved documents.
        """
        try:
            docsearch = Pinecone.from_existing_index(index, self.embeddings)

            if index == "eer-interaction-data":
                search_kwargs = {
                    "k": k,
                    "filter": {
                        "session_id": {"$ne": excluded_session_id}
                    }
                }
            else:
                search_kwargs = {"k": k}

            retriever = docsearch.as_retriever(search_kwargs=search_kwargs)
            docs = retriever.invoke(input)
            return docs
        except Exception as e:
            logger.error("Error retrieving documents: %s", e)
            raise

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
    def invoke_llm_with_retry(self, prompt):
        """
        Invokes the LLM with retry logic to handle transient connection issues.

        Args:
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The LLM's response to the prompt.
        """
        response = self.llm.invoke(prompt)
        return response

    def get_llm_response(self, prompt):
        """
        Generates a response from the LLM, retrying if necessary.

        Args:
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The LLM's response or an error message if the invocation fails.
        """
        try:
            response = self.invoke_llm_with_retry(prompt)
            return response
        except Exception as e:
            error = f"Error invoking LLM: {e}"
            return error

    def format_context(self, documents, chat=False):
        """
        Formats the context from retrieved documents for use in prompts.

        Args:
            documents (list): List of documents retrieved from Pinecone.
            chat (bool, optional): Whether to format as chat context. Defaults to False.

        Returns:
            str: A formatted string representing the context.
        """
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

    def upsert_vectorstore(self, user_input, ai_output, user_name, session_id):
        """
        Upserts the conversation data to the Pinecone vector store, storing user input, AI output, and metadata.

        Args:
            user_input (str): The user's input.
            ai_output (str): The AI's response.
            user_name (str): The name of the user.
            session_id (str): The unique session identifier.
        """
        try:
            pinecone_instance_chat = pc(api_key=os.getenv('PINECONE_API_KEY'), embeddings=self.embeddings)
            index_name = "eer-interaction-data"
            environment = "gcp-starter"
            index = pinecone_instance_chat.Index(index_name, environment=environment)

            embedding = self.embeddings.embed_documents([user_input + ai_output])[0]

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
            logger.info("Upsert to Pinecone successful for session_id: %s", session_id)
        except Exception as e:
            logger.error("Error upserting to Pinecone: %s", e)
            raise

    def pipeline(self, user_input, user_name, session_id, chat_history=None):
        """
        Handles the full pipeline of receiving user input, generating responses, and storing data.

        Args:
            user_input (str): The user's input.
            user_name (str): The name of the user.
            session_id (str): The unique session identifier.
            chat_history (str, optional): The chat history for context. Defaults to None.

        Returns:
            dict: A dictionary containing the AI output, source data, and past chat context.
        """
        if chat_history:
            chat_history = chat_history + "\n\n"
        else:
            chat_history = ""

        # Step 1: Retrieve source data
        source_data = self.retrieve_docs(user_input, "eer-transcripts-pdfs")
        formatted_source_data = self.format_context(source_data)

        # Step 2: Generate LLM response from source data
        sourcedata_response = self.get_llm_response(self.default_prompt_sourcedata(chat_history=chat_history, original_data=formatted_source_data, user_input=user_input, user_name=user_name))

        # Step 3: Retrieve past chat context
        past_chat_context = self.retrieve_docs(sourcedata_response, "eer-interaction-data", session_id)
        formatted_chat_context = self.format_context(past_chat_context, chat=True)

        # Step 4: Generate LLM response for conversation context, now considering combined chat history
        conversation_response = self.get_llm_response(self.default_prompt_conv(chat_history=chat_history, user_input=user_input, llm_response=sourcedata_response, past_chat=formatted_chat_context, user_name=user_name))

        # Step 5: Combine the responses
        ai_output = f"{sourcedata_response}\n\n{conversation_response}"
        logger.info(f"Pipeline generated response: {ai_output}")

        # Step 6: Upsert to vector store
        self.upsert_vectorstore(user_input, ai_output, user_name, session_id)

        # Return a dictionary containing all relevant information
        return {
            "ai_output": ai_output,
            "source_data": source_data,
            "past_chat_context": past_chat_context
        }
        
    @staticmethod
    def query_summaries(timestamp):
        # Set up pinecone client
        pineclient = pinecone.Pinecone(api_key=os.environ.get("2_PINECONE_API_KEY"))
        
        # Connect to the index
        index_name = "eer-meeting-summaries"
        index = pineclient.Index(index_name)
        
        # Convert to datetime object
        date_object = datetime.strptime(timestamp, "%Y-%m-%d")

        # Get the Unix timestamp
        unix_timestamp = int(date_object.timestamp())
        
        summary = index.query(vector = [1]*768, top_k=1,filter={"date_unix": {"$lte": unix_timestamp}}, include_metadata=True)
        
        return summary