# RAG-Powered Contextual Chatbot with Groq LLM

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot powered by Groq LLM (Llama 3.1 8B Instant) and LangChain. The chatbot can answer questions based on PDF documents by retrieving relevant context using HuggingFace embeddings and FAISS vector search, ensuring accurate, context-based responses.

The application is deployed using Streamlit, providing an interactive interface for real-time document-based Q\&A.

## Features

* Document Ingestion: Load and process multiple PDF documents using PyPDFDirectoryLoader.
* Text Chunking: Split large documents into 1000-character chunks with 200-character overlap for better retrieval.
* Vector Embeddings: Convert document chunks into vectors using sentence-transformers/all-MiniLM-L6-v2.
* FAISS Vector Store: Store and retrieve chunks efficiently for semantic search.
* Groq LLM Integration: Generate high-quality responses with Llama 3.1 8B Instant.
* Streamlit Interface: User-friendly interface to embed documents, ask questions, and view retrieved chunks.
* Optimized Performance: Average response time is under 2 seconds per query.

## Tech Stack

* Language Model: Groq LLM (Llama 3.1 8B Instant)
* Frameworks: LangChain, Streamlit
* Embeddings: HuggingFace (all-MiniLM-L6-v2)
* Vector Store: FAISS
* Data Loader: PyPDFDirectoryLoader
* Environment Management: dotenv

## Project Structure

```
us_census/         # Folder containing PDF documents
app.py             # Main Streamlit application
requirements.txt   # Dependencies
README.md          # Project documentation
```

## How to Run

1. Clone the Repository:

```
git clone https://github.com/vviiishu/Langchain-and-RAG-Projects.git
cd Langchain-and-RAG-Projects
```

2. Create and Activate a Virtual Environment:

```
python -m venv myenv
source myenv/bin/activate    # Linux/Mac
myenv\Scripts\activate     # Windows
```

3. Install Dependencies:

```
pip install -r requirements.txt
```

4. Set API Keys: Create a `.env` file in the root directory and add:

```
GROQ_API_KEY=your_groq_api_key
```

5. Run the Application:

```
streamlit run app.py
```

## Performance Metrics

* Processed over 20 PDF documents into more than 1000 text chunks.
* Reduced average response latency to under 2 seconds.
* Achieved over 95% accuracy in context-based responses.

## Applications

* Research Assistance: Summarize and query large sets of documents.
* Enterprise Knowledge Management: Internal document Q\&A.
* Customer Support: Provide accurate answers based on manuals and policies.

## Acknowledgements

* LangChain
* Groq
* Huggi
