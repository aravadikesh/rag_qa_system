# QASystem Project

## Overview

The QASystem project is a Question-Answering (QA) system that utilizes embeddings and a FAISS index to retrieve and generate answers based on a knowledge base. It can be run as a Flask server, accepting both CLI input and HTTP requests from clients.

## Features

- Load Q&A pairs from a specified knowledge file.
- Use a text generation model to generate answers based on user questions.
- Supports batch processing of questions.

You can install the required libraries using pip:

```bash
pip install -r requirements.txt

## Project Structure

QASystem/
├── QASystemModel.py      # Contains the QASystem class
├── server.py              # Flask server implementation
├── knowledge.txt          # Sample knowledge base (Q&A pairs)
├── questions.txt          # Sample questions file
├── client.py              # Client to interact with the server
└── README.md              # Project documentation


