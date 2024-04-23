import os
from typing import Dict, Any
from datetime import datetime, timedelta
from typing import List, Optional
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint as HuggingFaceHub
from langchain_community.vectorstores.timescalevector import TimescaleVector
from timescale_vector import client
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint as HuggingFaceHub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnablePassthrough,
    RunnableLambda,
)
from dotenv import find_dotenv, load_dotenv
load_dotenv()

_ = load_dotenv(find_dotenv())
huggingfacehub_api_token = os.environ['HUGGINGFACE_API_KEY']
SERVICE_URL = os.environ["timescale_url"]

class SimilarityRetriever(Runnable[Dict[str, Any], Dict[str, Any]]):
    
    def __init__(self, 
            db: TimescaleVector = None,
            embedding_model_name: str = "intfloat/multilingual-e5-large",
            collection_name: str = "eerallfiles",
            timescale_url: str = None):
        if db is None:
            db = TimescaleVector(
                    embedding=HuggingFaceEmbeddings(model_name=embedding_model_name),
                    collection_name=collection_name,
                    service_url=SERVICE_URL
                )
            self.db = db
        else:
            self.db = db
            
    def invoke(self, input: Dict[str, Any], config: Optional[Any] = None) -> Dict[str, Any]:
        query = input.get("question", "")
        start_date = input.get("start_date")
        end_date = input.get("end_date")
        k_value = config.get("k", 5) if config else 5

        # Check if the query is empty or not provided
        if query is None or query == "" or isinstance(query, str) and query.isspace():
            raise ValueError("Invalid or empty query provided")

        # Perform similarity search in database with query, start_date, and end_date
        context = self.db.similarity_search_with_score(query, start_date=start_date, end_date=end_date, k=k_value)

        # Return the context as part of the output dictionary
        return context

class ChatBot():    
    embeddings = HuggingFaceEmbeddings(model_name = "intfloat/multilingual-e5-large")
    SERVICE_URL = os.environ["timescale_url"]

    db = TimescaleVector(
        embedding=embeddings,
        collection_name="eerallfiles",
        service_url=SERVICE_URL
    )

    llm = None  # Define llm, it will be initialized in __init__

    print("HuggingFace Hub initialized...")

    default_template = """
    You are a chatbot working for the Experimenting Experiencing Reflecting (EER) Project, a research endeavor investigating the connections between art and science.
    You have access to a vast collection of documents, including research papers, meeting transcripts, and other relevant materials.
    The dialogue from the meetings may contain errors, prompting you to deduce the most probable information from the surrounding context.
    Your main task is to provide information and answer questions based on the available data to assist the project members in their tasks.
    All questions should pertain to the EER Project unless specified otherwise.
    In cases where questions mention "we," it is assumed to be referencing a member of the EER group.
    As the user engaging with you lacks knowledge of the provided context, ensure to provide relevant details for understanding 
    If uncertain about any details, kindly indicate so in your responses and maintain brevity in your answers.
    """
    
    template_end = """
    Context: {context}
    Question: {question}
    Answer: 
    
    """

    print("Prompt template created...")

    def __init__(self, custom_template=None, repo_id=None, temperature=0.8, k_value=1):  
        if custom_template:
            self.template = custom_template
        else:
            self.template = self.default_template
        
        self.temperature = temperature
        self.repo_id = repo_id
        self.k_value = k_value  # Store k_value
        
        # Initialize llm with repo_id
        self.llm = HuggingFaceHub(
            repo_id=self.repo_id,
            temperature=temperature,
            top_p=0.8,
            top_k=50,
            huggingfacehub_api_token=huggingfacehub_api_token
        )
    
        # Initialize the chain
        self.rag_chain = (
            {"context": SimilarityRetriever(self.db), "question": RunnablePassthrough()} 
            | PromptTemplate(template=self.template+self.template_end, input_variables=["context", "question"])
            | self.llm
        )
