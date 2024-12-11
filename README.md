<p align="left">
  <img src="https://static.vecteezy.com/system/resources/previews/024/673/126/original/question-answer-chat-document-paper-with-ai-artificial-intelligence-chat-bot-3d-render-icon-illustration-design-png.png" width="100" />
  <h1 align="left">EER (Experiencing, Experimenting, Reflecting) Chatbot</h1>
</p>

<p align="left">
    <img src="https://img.shields.io/github/license/sabszh/EER-chatbot-UI?style=flat&color=0080ff" alt="license">
    <img src="https://img.shields.io/github/last-commit/sabszh/EER-chatbot-UI?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
    <img src="https://img.shields.io/github/languages/top/sabszh/EER-chatbot-UI?style=flat&color=0080ff" alt="repo-top-language">
    <img src="https://img.shields.io/github/languages/count/sabszh/EER-chatbot-UI?style=flat&color=0080ff" alt="repo-language-count">
</p>
<p align="left">
		<em>Developed with the software and tools below.</em>
</p>
<p align="left">
	<img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white" alt="Streamlit">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/Pinecone-005BBB.svg?style=flat&logo=Python&logoColor=white" alt="Pinecone">
	<img src="https://img.shields.io/badge/HuggingFace-F79927.svg?style=flat&logo=HuggingFace&logoColor=white" alt="HuggingFace">
  <img src="https://img.shields.io/badge/Otter-005BBB.svg?style=flat&logo=Python&logoColor=white" alt="Otter">
</p>
<hr>

## Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot designed to assist with questions and exploration of meeting transcripts, summaries, and project data from the [EER](https://www.eer.info/) project. The chatbot integrates:
- **Document Retrieval**: Retrieves data from meeting transcripts, summaries, and related documents.
- **Conversation History**: References from past chatbot interactions.
- **LLM-Powered Summaries**: Uses a Large Language Model (LLM) to generate summaries of transcripts.

---

## File Structure

```
Project.
├── .venv                           # Virtual environment directory
├── data                            # Directory for storing input data (transcripts PDFs)
├── src                             # Source code directory
│    ├── preprocessimg
│    │    ├── reformatting_data.py  # Transcript reformatting scripts
│    │    └──data_chunking.py       # Data processing and chunking logic                
│    ├── streamlit_rag_chatbot      # Directory for TimescaleDB integration
│    │    ├── main.py               # Core chatbot pipeline
│    │    └── streamlit_app.py      # Streamlit app for the chatbot  
│    └── upserting_transcripts      # Directory for managing TimescaleDB integration
│         ├── a2t.py                # Incomplete script for the pipeline; currently focuses on adding transcripts to Pinecone
│         └── streamlit_a2t.py      # Streamlit app interface for managing the pipeline
├── .env                            # Environment variables (API keys for HuggingFace and Pinecone)
├── .gitignore                      # Excluded files and directories
├── Dockerfile                      # Docker configuration for deploying the app
├── LICENSE.txt                     # License for the project
├── README.md                       # Readme file
└── requirements.txt                # Dependencies for the project
 
```

---

## Modules

### Core Components

| File                                   | Summary                                                                                                                                                     |
|---------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`src/streamlit_rag_chatbot/main.py`**                     | Sets up the chatbot pipeline, integrating document retrieval with Pinecone and HuggingFace embeddings for advanced querying and summarization.              |
| **`src/streamlit_rag_chatbot/streamlit_app.py`**            | Implements the Streamlit-based user interface, enabling interaction with the chatbot and meeting summaries and referenced data.           |
| **`src/preprocessing/reformatting_data.py`**        | Automates cleaning and reformatting raw transcript files into a structured format (CSV), making them suitable for further processing.                      |
| **`src/preprocessing/data_chunking.py`**            | Splits transcripts into manageable chunks and prepares them for storage in the vector store with metadata enrichment.                                       |

---

## Getting Started

### Prerequisites

- Python 3.x.x
- API keys for:
  - HuggingFace
  - Pinecone

### Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/sabszh/EER-chatbot-UI/
    ```

2. **Navigate to the project directory**:

    ```bash
    cd EER-chatbot-UI
    ```

3. **Set up a virtual environment** (optional but recommended):

    ```bash
    python -m venv .venv
    source .venv/bin/activate   # On Windows: .venv\Scripts\activate
    ```

4. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

5. **Configure environment variables**:
    Create a `.env` file in the root directory and add your API keys:

    ```plaintext
    HUGGINGFACE_API_KEY=your_huggingface_api_key
    PINECONE_API_KEY=your_pinecone_api_key
    ```

### Running the Application

#### Using the Streamlit App

To launch the Streamlit app:

```bash
streamlit run src/streamlit_rag_chatbot/streamlit_app.py
```

#### Using Docker

1. **Build the Docker image**:
    ```bash
    docker build -t eer-chatbot .
    ```

2. **Run the Docker container**:
    ```bash
    docker run -p 8501:8501 eer-chatbot
    ```

---

## Features

### **Meeting Summary Fetching**
Fetch concise summaries of past meetings, filtered by specific dates. Summaries highlight discussion points, action items, and speaker lists.

### **Referenced Data Display**
View data sources referenced by the chatbot in its answers, including meeting transcripts and related documents.

### **Conversation History Integration**
Explore connections between your queries and those from other users, using past conversations for context-aware insights.

---

## License

This project is licensed under the GNU General Public License (GPL). 

TL;DR
1. Anyone can copy, modify and distribute this software.
2. You have to include the license and copyright notice with each and every distribution.
3. You can use this software privately.
4. You can use this software for commercial purposes.
5. If you dare build your business solely from this code, you risk open-sourcing the whole code base.
6. If you modify it, you have to indicate changes made to the code.
7. Any modifications of this code base MUST be distributed with the same license, GPLv3.
8. This software is provided without warranty.
9. The software author or license can not be held liable for any damages inflicted by the software.
- Access the full license text in the [LICENSE.txt](./LICENSE.txt) file.

For more details on the terms of this license, please visit [GNU Licenses](https://www.gnu.org/licenses/gpl-3.0.en.html).
