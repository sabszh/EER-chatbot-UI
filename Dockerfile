# Use the official Python image as base
FROM python:3.11

# Set the working directory
WORKDIR /EER-chatbot-UI

# Copy the requirements file
COPY requirements.txt .

# Install CPU-only version of PyTorch and torchvision
RUN pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install sentence-transformers

# Copy the rest of the application code
COPY . /EER-chatbot-UI

# Set the entry point for streamlit
ENTRYPOINT ["streamlit", "run", "src/streamlit_rag_chatbot/streamlit_app.py", \
            "--server.port=80", \
            "--server.headless=true", \
            "--server.address=0.0.0.0", \
            "--browser.gatherUsageStats=false", \
            "--server.enableStaticServing=true", \
            "--server.fileWatcherType=none", \
            "--client.toolbarMode=viewer"]