from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def datachunk():
    transcripts_folder = os.path.join("..","data","reformatted_transcripts")
    pdf_folder = os.path.join("..","data","EER-site-pages-pdf")

    print("Loading files...")

    txt_files = [file for file in os.listdir(transcripts_folder) if file.endswith('.txt')]
    pdf_files = [file for file in os.listdir(pdf_folder) if file.endswith('.pdf')]

    documents = []

    # Load text documents from .txt files
    for txt_file in txt_files:
        loader = TextLoader(os.path.join(transcripts_folder, txt_file))
        documents.extend(loader.load())

    print("Text files loaded...")

    # Load text documents from .pdf files
    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
        documents.extend(loader.load())

    print("PDF files loaded...")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, length_function=len)
    docs = text_splitter.split_documents(documents)
    
    return docs