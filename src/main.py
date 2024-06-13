from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint as HuggingFaceHub
from dotenv import load_dotenv
import os

load_dotenv()

class ChatBot():
    def __init__(self, custom_template=None, repo_id=None, temperature=0.8):
        self.embeddings = HuggingFaceEmbeddings()
        self.index_name = "eerbot"
        self.docsearch = Pinecone.from_existing_index(self.index_name, self.embeddings)

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
            {"context": multiquery_retriever_llm, "question": RunnablePassthrough()}
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