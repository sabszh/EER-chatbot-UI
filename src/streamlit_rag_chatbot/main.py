<<<<<<< Updated upstream:src/main.py
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
=======
import os
from datetime import datetime, timezone
import logging
from tenacity import retry, stop_after_attempt, wait_fixed
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint as HuggingFaceHub
>>>>>>> Stashed changes:src/streamlit_rag_chatbot/main.py
from langchain_community.vectorstores.pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint as HuggingFaceHub
from dotenv import load_dotenv
import os
from data_chunking import datachunk
from pinecone import Pinecone as pc
from pinecone import PodSpec
from pinecone import ServerlessSpec

<<<<<<< Updated upstream:src/main.py
load_dotenv()

class ChatBot():
    def __init__(self, custom_template=None, repo_id=None, temperature=0.8):
        self.embeddings = HuggingFaceEmbeddings()
        self.index_name = "boteer"
        pinecone_instance = pc(api_key=os.getenv('PINECONE_API_KEY'), embeddings=self.embeddings)
        spec = ServerlessSpec(cloud="aws",region="us-east-1")
        
        #if self.index_name not in pinecone_instance.list_indexes().names():
        #    docs = datachunk()
        #    pinecone_instance.create_index(name=self.index_name, metric="cosine", dimension=768, spec=spec)
        #    self.docsearch = Pinecone.from_documents(docs, self.embeddings, index_name=self.index_name)
        #    print("Created new Pinecone index and loaded documents")
        #else:
        #    self.docsearch = Pinecone.from_existing_index(self.index_name, self.embeddings)
        #    print("Using existing Pinecone index")
    
        self.docsearch = Pinecone.from_existing_index(self.index_name, self.embeddings)
=======
# Set up logging for the chatbot
logger = logging.getLogger(__name__)

# The chatbot class definition with methods for RAG-based document retrieval and LLM-based conversation generation
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
        self.pinecone_instance = pc(api_key=os.getenv('PINECONE_API_KEY_2'), embeddings=self.embeddings)
>>>>>>> Stashed changes:src/streamlit_rag_chatbot/main.py

        self.template = custom_template if custom_template else self.default_template()
        self.temperature = temperature
        self.repo_id = repo_id
        
        self.llm = HuggingFaceHub(
            repo_id=self.repo_id,
            temperature=temperature,
            top_p=0.8,
            top_k=50,
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )

        multiquery_retriever_llm = MultiQueryRetriever.from_llm(retriever=self.docsearch.as_retriever(), llm=self.llm)

        # Initialize the chain
        self.rag_chain = (
            {"context": multiquery_retriever_llm.batch, "question": RunnablePassthrough()}
            | PromptTemplate(template=self.template + self.template_end(), input_variables=["context", "question"])
            | self.llm
            | StrOutputParser()
        )

        print("Chain assembled...")

    def default_template(self):
        return """
        You are a chatbot working for the Experimenting Experiencing Reflecting (EER) Project, a research endeavor investigating the connections between art and science.
        You have access to a collection of documents, including descriptions of research activities, meeting transcripts, and other relevant materials.
        Your main task is to help the user explore and reflect on the EER project.
        All questions should pertain to the EER Project unless specified otherwise.
        When possible, please cite source documents in your answer (calling the documents the transcript date, e.g. "2021-05-28", or website page after the "/").
        """
    
    def template_end(self):
        return """
        Context: {context}
        Question: {question}
        Answer: 
        """
        
<<<<<<< Updated upstream:src/main.py
if __name__ == "__main__":
    ChatBot(repo_id="mistralai/Mistral-7B-Instruct-v0.2")
    print("Chatbot initialized...")
=======
    @staticmethod
    def query_summaries(timestamp):
        # Set up pinecone client
        pineclient = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY_2"))
        
        # Connect to the index
        index_name = "eer-meetings-summaries"
        index = pineclient.Index(index_name)
        
        # Example vector query (should be replaced with actual vector for filtering)
        query_vector = [1] * 768  # Replace with meaningful vector logic
        
        # Query the index
        results = index.query(
            vector=query_vector,
            top_k=100,
            include_metadata=True
        )
        
        results_dict = results.to_dict()
        
        for match in results_dict["matches"]:
            if match["metadata"]["date"] == timestamp:
                return match["metadata"]
        return "No match found"
>>>>>>> Stashed changes:src/streamlit_rag_chatbot/main.py
