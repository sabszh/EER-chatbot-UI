import streamlit as st
from datetime import datetime
import pandas as pd
from a2t import TranscriptProcessor  

# Initialize the TranscriptProcessor
processor = TranscriptProcessor()

st.title("Transcript Upsert Demo")
st.write("Upload one or multiple transcript CSV files to process and upsert summaries into Pinecone.")

# File uploader for single or multiple files
uploaded_files = st.file_uploader("Choose CSV files", accept_multiple_files=True, type="csv")

# Process and upsert when the user clicks the button
if uploaded_files and st.button("Process and Upsert to Pinecone"):
    with st.spinner("Processing transcripts..."):
        # Process the transcripts
        transcript_data = processor.process_transcripts(uploaded_files)
        
        # Display summaries to the user
        st.subheader("Generated Summaries")
        for filename, data in transcript_data.items():
            st.write(f"**File:** {filename}")
            st.write(f"**Speakers:** {', '.join(data['speakers'])}")
            st.write(f"**Summary:** {data['summary']}")
        
        # Upsert the summaries to Pinecone
        processor.upsert_summaries_to_pinecone(transcript_data)
        
        st.success("Summaries have been upserted to Pinecone!")
