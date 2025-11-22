# AI Medical RAG Chatbot

This project is an end-to-end Retrieval-Augmented Generation (RAG) medical chatbot built with LangChain, Google FLAN-T5, FAISS, and a lightweight Flask frontend. The system loads medical PDFs, converts them into embeddings, indexes them in a vector store, and answers queries using grounded context retrieved from the documents.

---

## Features

- PDF ingestion and preprocessing
- Text chunking with LangChain
- Embedding generation and FAISS vector indexing
- Google FLAN-T5 for answer generation
- Custom retriever pipeline
- Flask backend serving chat responses
- Simple HTML/CSS chat UI
- Environment-variable based configuration
- Optional Docker + CI/CD deployment workflow

---

## System Architecture Overview

### 1. Project Setup and Configuration
- Initialize project structure
- Set up virtual environment and logging
- Custom exception handling
- Configuration module to manage environment variables

### 2. Data Processing and Vector Storage
- Load medical PDFs
- Chunk text using LangChain
- Generate embeddings
- Store embeddings in FAISS index
- Data loader merges PDF loader, embeddings, vector store

### 3. LLM and Retrieval
- Set up Google FLAN-T5 model
- Implement retriever to query FAISS index
- Build application-level retrieval wrapper

### 4. Application Layer
- Flask backend routing and chat logic
- Clean HTML/CSS UI for user interaction

### 5. Deployment (Optional)
- GitHub versioning
- Dockerfile for container packaging
- Jenkins CI/CD setup
- Push images to AWS ECR
- Deploy to AWS runner

---

## Tech Stack

### Core
- Python 3.10+
- Flask
- LangChain
- FAISS
- Google FLAN-T5

### Frontend
- HTML + CSS

### DevOps (Optional)
- Docker
- Jenkins
- AWS ECR / AWS Run

---

## Folder Structure

```
Medical-RAG-Chatbot/
│
├── app/
│   ├── components/
│   │   ├── llm.py
│   │   ├── retriever.py
│   │   ├── vector_store.py
│   │   ├── embeddings.py
│   │   ├── pdf_loader.py
│   │   └── data_loader.py
│   │
│   ├── config/
│   │   ├── config.py
│   │   └── __init__.py
│   │
│   ├── common/
│   │   ├── exceptions.py
│   │   └── logger.py
│   │
│   ├── app.py
│   └── templates/
│       └── index.html
│
├── static/
├── docker/
├── requirements.txt
└── README.md
```

---

## Future Enhancements
- Add streaming answers
- Add React-based UI
- Support multimodal RAG
- Add session-based memory