<p align="left">
  <img src="https://static.vecteezy.com/system/resources/previews/024/673/126/original/question-answer-chat-document-paper-with-ai-artificial-intelligence-chat-bot-3d-render-icon-illustration-design-png.png" width="100" />
  <h1 align="left">EER (Experiencing, Experimenting, Reflecting) Chatbot</h1>
</p>

<p align="left">
    <img src="https://img.shields.io/github/license/sabszh/EER-chatbot-UI?style=flat&color=0080ff" alt="license">
    <img src="https://img.shields.io/github/last-commit/sabszh/EER-chatbot-UI?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
    <img src="https://img.shields.io/github/languages/top/sabszh/EER-chatbot-UI?style=flat&color=0080ff" alt="repo-top-language">
    <img src="https://img.shields.io/github/languages/count/sabszh/EER-chatbot-UI?style=flat&color=0080ff" alt="repo-language-count">
<p>
<p align="left">
		<em>Developed with the software and tools below.</em>
</p>
<p align="left">
	<img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white" alt="Streamlit">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
</p>
<hr>

##  Overview

This project implements a chatbot utilizing retrieval mechanisms (RAG) to serve as a question-answering assistant. It leverages excerpts from transcripts of Zoom meetings pertaining to the [EER](https://www.eer.info/) project.

NB: For timescale version of app, see here: https://github.com/sabszh/EER-TIMEBOT

---

##  Repository Structure

```sh
└── /

    ├── README.md
    ├── requirements.txt
    └── src
        ├── data_chunking.py
        ├── main.py
        ├── reformatting_data.py
        ├── streamlit_app.py
        └── timescale
```

---

##  Modules

<details closed><summary>.</summary>

| File                                                                                      | Summary                                                                                                                                                                                        |
| ---                                                                                       | ---                                                                                                                                                                                            |
| [requirements.txt](https://github.com/sabszh/EER-chatbot-UI/blob/master/requirements.txt) | This `requirements.txt` ensures the application's compatibility and functionality by defining necessary Python packages for the data processing and web application modules of the repository. |                                 |

</details>

<details closed><summary>src</summary>

| File                                                                                                  | Summary                                                                                                                                                                                                          |
| ---                                                                                                   | ---                                                                                                                                                                                                              |
| [main.py](https://github.com/sabszh/EER-chatbot-UI/blob/master/src/main.py)                           | The `main.py` within this repository sets up a chatbot leveraging Pinecone index and HuggingFace embeddings for document search, with custom prompt templates for the EER Project's inquiries. |
| [streamlit_app.py](https://github.com/sabszh/EER-chatbot-UI/blob/master/src/streamlit_app.py)         | Core interface of the chatbot, allowing user to select AI models and adjust interaction parameters within a Streamlit-based web app.                                                                               |
| [reformatting_data.py](https://github.com/sabszh/EER-chatbot-UI/blob/master/src/reformatting_data.py) | The script `reformatting_data.py` within the repo transforms raw transcript files into a structured CSV format, handling various timestamp styles and creating a standardized naming and storage schema.         |
| [data_chunking.py](https://github.com/sabszh/EER-chatbot-UI/blob/master/src/data_chunking.py)         | The data_chunking.py module is responsible for extracting and preparing text data from various document types within a document processing pipeline.                                                             |

</details>

---

##  Getting Started

***Requirements***

Ensure you have the following dependencies installed on your system:

* **Python**: `version 3.x.x`
* **API Keys**: Obtain API keys for Hugging Face, Pinecone, and/or TimescaleDB.

###  Installation

1. Clone the  repository:

```sh
git clone https://github.com/sabszh/EER-chatbot-UI/
```

2. Change to the project directory:

```sh
cd 
```

3. Install the dependencies:

```sh
pip install -r requirements.txt
```

4. Create a .env file in the root directory of the project and add the following:

```sh
HUGGINGFACE_API_KEY=your_huggingface_api_key
PINECONE_API_KEY=your_pinecone_api_key
TIMESCALE_API_KEY=your_timescale_api_key
```

###  Running 

Use the following command to run streamlit app locally using Pinecone index:

```sh
streamlit run src/streamlit_app.py
```

Use the following command to run streamlit app locally using Timescale index:

```sh
streamlit run src/timescale/streamlit_app_ts.py
```