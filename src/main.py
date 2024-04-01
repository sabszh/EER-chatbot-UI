from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint as HuggingFaceHub
from dotenv import load_dotenv
from pinecone import Pinecone as pc
from pinecone import PodSpec
import os

load_dotenv()

class ChatBot():
    transcripts_folder = 'data/reformatted_transcripts/'
    pdf_folder = 'data/EER-site-pages-pdf/'
    
    txt_files = [file for file in os.listdir(transcripts_folder) if file.endswith('.txt')]
    pdf_files = [file for file in os.listdir(pdf_folder) if file.endswith('.pdf')]

    documents = []

    # Load text documents from .txt files
    for txt_file in txt_files:
        loader = TextLoader(os.path.join(transcripts_folder, txt_file))
        documents.extend(loader.load())
        
    # Load text documents from .pdf files
    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, length_function = len)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    pinecone_instance = pc(api_key=os.getenv('PINECONE_API_KEY'), embeddings = embeddings)

    index_name = "eerbot"
    environment = "gcp-starter"
    spec = PodSpec(environment=environment)
    
    if index_name not in pinecone_instance.list_indexes():
        pinecone_instance.create_index(name=index_name, dimension=768, metric="cosine", spec=spec)
        docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    else:
        docsearch = pc.from_existing_index(index_name, embeddings)


    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    
    llm = HuggingFaceHub(
        repo_id=repo_id,
        temperature=0.8,
        top_p=0.8,
        top_k=50,
        huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
    )

    from langchain.prompts import PromptTemplate 

    template = """
    You are a creative expert on the Experimenting Experiencing Reflecting (EER) Project, a research initiative exploring the intersections between art and science.
    Your task is to respond to questions using details from excerpts taken from meetings within a research group focusing on the EER Project.
    The transcripts from the meetings may contain errors, and you should retrieve the most likely information from the context.
    Assume all questions are related to the EER Project.
    When questions refer to "we," assume it is from a member of the EER group.
    Only use information from the provided document excerpts to respond, you are allowed to have a bit of creative freedom but keep it within the relevant domains.
    If uncertain, indicate so. Keep your answers concise.

    Context: {context}
    Question: {question}
    Answer: 

    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser

    rag_chain = (
        {"context": docsearch.as_retriever(),  "question": RunnablePassthrough()} 
        | prompt 
        | llm
        | StrOutputParser() 
    )
