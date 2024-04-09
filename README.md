# EER (Experiencing, Experimenting, Reflecting) Chatbot
## Project Description
This project implements a chatbot utilizing retrieval mechanisms (RAG) to serve as a question-answering assistant. It leverages excerpts from transcripts of Zoom meetings pertaining to the [EER](https://www.eer.info/) project.

## Overview

The chatbot consists of three main files:
- `data_chunking.py`: Defines functions for loading and chunking text data from text and PDF files.
- `main.py`: Contains the core logic of the chatbot, including initialization of language models, document loading, RAG chain and response generation.
- `streamlit_app.py`: Implements the Streamlit web application for interacting with the chatbot.

**Setup Instructions:**
1. Clone the repository containing the project files.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Ensure you have API keys for Pinecone and Hugging Face, and set them as environment variables in a `.env` file.
4. Organize your data into the specified folders:
   - Place text documents in the 'data/reformatted_transcripts/' folder.
   - Place PDF documents in the 'data/EER-site-pages-pdf/' folder.
5. Run `python main.py` to initialize the chatbot and create the Pinecone index.
6. Launch the Streamlit app by running `streamlit run streamlit_app.py`.
7. Interact with the chatbot through the Streamlit user interface.

## Setup

To run the chatbot, follow these steps:

1. Ensure you have Python installed on your system.
2. Install the required dependencies by running:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - `PINECONE_API_KEY`: API key for Pinecone indexing service.
   - `HUGGINGFACE_API_KEY`: API key for accessing Hugging Face models.

**Usage:**
1. Run `main.py` to initialize the chatbot and create the Pinecone index.
2. Launch the Streamlit app by running `streamlit run streamlit_app.py`.
4. Interact with the chatbot by providing prompts and observing its responses in the Streamlit interface.

## Dependecies
Make sure to add data in the data folder.

## Repo structure

```plaintext
Project Root
│
├── .venv
├── .env
├── .devcontainer
│   └── devcontainer.json
│
├── data
│   └── .folders
│
├── src
│   ├── data_chunking.py
│   ├── main.py
│   ├── reformatting_data.py
│   └── streamlit_app.py
│
├── .gitignore
├── README.md
└── requirements.txt
```
