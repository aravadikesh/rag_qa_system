# QASystem Project

## Overview

The QASystem project is a Question-Answering (QA) system that utilizes embeddings and a FAISS index to retrieve and generate answers based on a knowledge base. It can be run as a Flask-ML server, accepting both CLI input and HTTP requests from clients.

## Features

- Load Q&A pairs from a specified knowledge file.
- Use a text generation model to generate answers based on user questions.
- Supports batch processing of questions.

## Project Structure

QASystem/
├── QASystemModel.py      # Contains the QASystem class
├── server.py              # Flask server implementation
├── knowledge.txt          # Sample knowledge base (Q&A pairs)
├── questions.txt          # Sample questions file
├── client.py              # Client to interact with the server
└── README.md              # Project documentation

## Setup

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

You can install the required libraries using pip:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Command Line Interface (CLI)

You can run the QA system directly from the command line using the following command:

```bash
python python QA_system.py --knowledge_path knowledge.txt --questions_path questions.txt --prompt "What is NumPy?"
```
--knowledge_path: Path to the file containing Q&A pairs.
--questions_path: Path to the file containing questions to be answered.
--prompt: An optional question to ask directly via CLI.

### 2. Flask-ML

```bash
python server.py --knowledge_path knowledge.txt --questions_path questions.txt
```

In a separate terminal window, you can interact with the server using the client.py script. You can specify the inputs to be sent to the server as shown below:

```python 
inputs = [
    {"text": "What is NumPy?"}
]
```

Then run the client via the following:

```bash
python client.py
```








