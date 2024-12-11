import os
import pandas as pd
from datetime import datetime
import time
import re
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint as HuggingFaceHub
from langchain_community.vectorstores.pinecone import Pinecone
from pinecone import Pinecone as pc
import pinecone

load_dotenv()

class TranscriptProcessor:
    def __init__(self, model_repo="meta-llama/Llama-3.1-70B-Instruct", temperature=1.0, api_token=None):
        self.llm = HuggingFaceHub(
            repo_id=model_repo,
            temperature=temperature,
            huggingfacehub_api_token=api_token or os.getenv('HUGGINGFACE_API_KEY')
        )
        self.embeddings = HuggingFaceEmbeddings()
        self.pinecone_instance_chat = pc(api_key=os.getenv('PINECONE_API_KEY_2'), embeddings=self.embeddings)
    
    def summary_prompt(self, text):
        return f"Please identify the main discussion points, decisions, and action items from my meeting notes below and provide a concise bulleted summary. Here is the meeting transcript: '{text}'. Your notes and summary:"
    
    def extract_unique_speakers(self, file):
        """Extract unique speakers from an uploaded CSV file."""
        try:
            df = pd.read_csv(file, delimiter=";", encoding="ISO-8859-1")
            unique_speakers = df['speaker_name'].unique().tolist()
            return unique_speakers
        except Exception as e:
            print(f"Error extracting speakers from file: {e}")
            return []

    def load_csv_content(self, file):
        """Load the content of the uploaded CSV file for summarization."""
        try:
            file.seek(0)  # Ensure the file pointer is at the beginning
            return file.read()  # No need to decode, just read the content
        except Exception as e:
            print(f"Error loading file content: {e}")
            return ""

    def process_transcripts(self, files):
        """Process a single uploaded file or multiple uploaded files and return a dictionary of transcript data."""
        transcript_data = {}

        if isinstance(files, list):
            for file in files:
                if file.name.endswith('.csv'):
                    speakers = self.extract_unique_speakers(file)
                    transcript_text = self.load_csv_content(file)
                    summary = self.llm.invoke(self.summary_prompt(transcript_text))
                    transcript_data[file.name] = {
                        "speakers": speakers,
                        "summary": summary
                    }
        
        else:
            if files.name.endswith('.csv'):
                speakers = self.extract_unique_speakers(files)
                transcript_text = self.load_csv_content(files)
                summary = self.llm.invoke(self.summary_prompt(transcript_text))
                transcript_data[files.name] = {
                    "speakers": speakers,
                    "summary": summary
                }

        return transcript_data

    def upsert_summaries_to_pinecone(self, data):
        """Upsert processed summaries to Pinecone."""
        index_name = "eer-meetings-summaries"
        environment = "gcp-starter"
        index = self.pinecone_instance_chat.Index(index_name, environment=environment)

        for i, (meeting_id, meeting_data) in enumerate(data.items(), start=1):
            summary = meeting_data['summary']
            speakers = meeting_data['speakers']

            # Create embedding for the summary
            embedding = self.embeddings.embed_documents([summary])[0]

            # Use regex to find the date at the beginning of the filename
            match = re.match(r"(\d{4}-\d{2}-\d{2})", meeting_id)
            if match:
                date_str = match.group(1)
                try:
                    dt_obj = datetime.strptime(date_str, "%Y-%m-%d")
                    unix_timestamp = int(time.mktime(dt_obj.timetuple()))
                except ValueError as e:
                    print(f"Error parsing date from filename '{meeting_id}': {e}")
                    continue
            else:
                print(f"No valid date found in filename '{meeting_id}'")
                continue

            # Upsert to Pinecone
            index.upsert(vectors=[
                {
                    'id': f"{unix_timestamp}",
                    'values': embedding,
                    'metadata': {
                        "date": date_str,
                        "summary": summary,
                        "speakers": speakers,
                        "date_unix": unix_timestamp,
                        "text": summary
                    }
                }
            ])
