from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import pandas as pd

def datachunk(transcripts_folder, pdf_folder):
    """
    Load text documents from CSV and PDF files and split them into chunks.
    """
    csv_files = [file for file in os.listdir(transcripts_folder) if file.endswith('.csv')]
    pdf_files = [file for file in os.listdir(pdf_folder) if file.endswith('.pdf')]

    documents_csv = []
    documents_pdf = []

    # Extract text documents and add metadata from CSV files
    for csv_file in csv_files:
        try:
            transcripts_df = pd.read_csv(os.path.join(transcripts_folder, csv_file), delimiter=';')
        except FileNotFoundError:
            print(f"Error: The file {csv_file} was not found.")
            continue

        for index, row in transcripts_df.iterrows():
            document = Document(
                page_content=row["transcript_text"],
                metadata={
                    "speaker_name": row["speaker_name"],
                    "date_time": row["date_time"],
                    "source": csv_file
                }
            )
            documents_csv.append(document)

    print("CSV files loaded...")

    # Load text documents from .pdf files (keeping as they are)
    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
        doc_content = loader.load()
        documents_pdf.extend(doc_content)

    print("PDF files loaded...")

    
    # Split text documents into chunks
    text_documents = [doc for doc in documents_pdf if isinstance(doc, Document) and doc.page_content]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, length_function=len, )
    split_docs = text_splitter.split_documents(text_documents)

    # Combine split text documents with PDF documents
    docs = split_docs + [doc for doc in documents_pdf if not isinstance(doc, Document)]
    
    # Add data from CSV files to the combined documents
    docs.extend(documents_csv)
    
    return docs