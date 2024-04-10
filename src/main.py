from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint as HuggingFaceHub
from dotenv import load_dotenv
from pinecone import Pinecone as pc
from pinecone import PodSpec
import os

from data_chunking import datachunk

load_dotenv()

class ChatBot():
    
    embeddings = HuggingFaceEmbeddings()
    
    pinecone_instance = pc(api_key=os.getenv('PINECONE_API_KEY'), embeddings=embeddings)
    
    index_name = "eerbot"
    environment = "gcp-starter"
    spec = PodSpec(environment=environment)
    
    if index_name not in pinecone_instance.list_indexes().names():
        docs = datachunk()
        pinecone_instance.create_index(name=index_name, metric="cosine", dimension=768, spec=spec)
        docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
        print("Created new Pinecone index and loaded documents")
    else:
        docsearch = Pinecone.from_existing_index(index_name, embeddings)
        print("Using existing Pinecone index")
    
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    llm = HuggingFaceHub(
        repo_id=repo_id,
        temperature=0.8,
        top_p=0.8,
        top_k=50,
        huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
    )

    print("HuggingFace Hub initialized...")

    from langchain.prompts import PromptTemplate

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

    def __init__(self, custom_template=None):
        if custom_template:
            self.template = custom_template
        else:
            self.template = self.default_template
    
        self.rag_chain = (
            {"context": self.docsearch.as_retriever(), "question": RunnablePassthrough()}
            | PromptTemplate(template=self.template+self.template_end, input_variables=["context", "question"])
            | self.llm
            | StrOutputParser()
        )

    print("Chain assembled...")
