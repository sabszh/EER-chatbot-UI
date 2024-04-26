""" from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime, timedelta
from timescale_vector import client
import os

def datachunk(transcripts_folder,pdf_folder):
    def create_uuid(date_string: str):
        if date_string is None:
            return None
        time_format = '%a %b %d %H:%M:%S %Y'
        date_string = date_string.strip()  # Remove leading and trailing whitespace
        datetime_obj = datetime.strptime(date_string, time_format)
        uuid = client.uuid_from_time(datetime_obj)
        return str(uuid)
    
    csv_files = [file for file in os.listdir(transcripts_folder) if file.endswith('.csv')]
    #pdf_files = [file for file in os.listdir(pdf_folder) if file.endswith('.pdf')]

    documents = []

    # Load text documents from .txt files
    for csv_file in csv_files:
        loader = CSVLoader(
            os.path.join(transcripts_folder, csv_file),
            csv_args={"delimiter": ";"},
            metadata_columns=["speaker_name", "date_time"])
        loaded_documents = loader.load()
        
        for doc in loaded_documents:
            date_string = doc.metadata.get("date_time") 
            if date_string:
                uuid = create_uuid(date_string)
                doc.metadata["uuid"] = uuid
                
        documents.extend(loaded_documents)

    print("Text files loaded...")

    # Load text documents from .pdf files
    #for pdf_file in pdf_files:
        #loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
        #documents.extend(loader.load())

    #print("PDF files loaded...")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    return docs
    """