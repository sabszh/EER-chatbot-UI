from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint as HuggingFaceHub
from dotenv import load_dotenv
from pinecone import Pinecone as pc
from pinecone import PodSpec
import os

from data_chunking import datachunk

load_dotenv()

class ChatBot():
    
    embeddings = HuggingFaceEmbeddings()
    print("Embeddings initialized...")
    
    pinecone_instance = pc(api_key=os.getenv('PINECONE_API_KEY'), embeddings=embeddings)
    
    index_name = "eerbot"
    environment = "gcp-starter"
    spec = PodSpec(environment=environment)
    
    if index_name not in pinecone_instance.list_indexes().names():
        print("Index does not exist, creating...")
        docs = datachunk()
        print("Data chunked...")
        pinecone_instance.create_index(name=index_name, metric="cosine", dimension=768, spec=spec)
        print("Index created...")
        print("Loading documents into Pinecone index...")
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

    template = """
    You are a creative expert on the Experimenting Experiencing Reflecting (EER) Project, a research initiative exploring the intersections between art and science.
    Your task is to respond to questions using details from excerpts taken from meetings within a research group focusing on the EER Project.
    The transcripts from the meetings may contain errors, and you should retrieve the most likely information from the context.
    Assume all questions are related to the EER Project unless otherwise stated.
    When questions refer to "we," assume it is from a member of the EER group.
    If uncertain, indicate so. Keep your answers concise.

    Context: {context}
    Question: {question}
    Answer: 

    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    print("Prompt template created...")

    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser

    rag_chain = (
            {"context": docsearch.as_retriever(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    print("Chain assembled...")
