
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import pandas as pd

def datachunk():
    transcripts_folder = os.path.join("data", "reformatted_transcripts")
    pdf_folder = os.path.join("data", "EER-site-pages-pdf")

    print("Loading files...")

    csv_files = [file for file in os.listdir(transcripts_folder) if file.endswith('.csv')]
    pdf_files = [file for file in os.listdir(pdf_folder) if file.endswith('.pdf')]

    documents = []

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
            documents.append(document)

    print("CSV files loaded...")

    # Load text documents from .pdf files (keeping as they are)
    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
        doc_content = loader.load()
        documents.extend(doc_content)

    print("PDF files loaded...")

    # Split text documents into chunks
    text_documents = [doc for doc in documents if isinstance(doc, Document) and doc.page_content]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, length_function=len)
    split_docs = text_splitter.split_documents(text_documents)

    # Combine split text documents with PDF documents
    docs = split_docs + [doc for doc in documents if not isinstance(doc, Document)]

    return docs