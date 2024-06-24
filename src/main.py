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
from data_chunking import datachunk
from pinecone import Pinecone as pc
from pinecone import PodSpec
from pinecone import ServerlessSpec

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
        
if __name__ == "__main__":
    ChatBot(repo_id="mistralai/Mistral-7B-Instruct-v0.2")
    print("Chatbot initialized...")