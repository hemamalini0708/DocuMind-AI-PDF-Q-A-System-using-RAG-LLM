# 📄 DocuMind AI — Intelligent Document Q&A System

> Upload any PDF. Ask anything. Get precise answers with source page references.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.2-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.6-orange)
![Groq](https://img.shields.io/badge/Groq-LLaMA3-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-1.43-red)

---

## 🧠 What is DocuMind AI?

DocuMind AI is a production-grade **Retrieval-Augmented Generation (RAG)** system
that allows users to upload any PDF document and ask questions about it in plain English.

Instead of reading through hundreds of pages manually, DocuMind AI:
- Reads and understands your document
- Finds the most relevant sections for your question
- Generates a precise, context-aware answer
- Shows exactly which pages the answer came from

Tested on real documents including research papers, datasets books,
corporate reports (160 pages), and resumes — all with accurate results.

---

## 🔍 Problem It Solves

| Old Way | DocuMind AI Way |
|---|---|
| Read 160 pages manually | Upload and ask in seconds |
| Ctrl+F for keywords | Ask in plain English |
| Miss context across pages | AI understands full document |
| No source tracking | Shows exact page numbers |

---

## 🏗️ RAG Architecture
```
PDF Upload

↓

Document Loader   → Extracts text page by page (PyPDFLoader)

↓

Text Chunker      → Splits into 1000-char overlapping chunks

↓

Embedding Model   → Converts chunks to 384-dim vectors (HuggingFace)

↓

Vector Store      → Saves all vectors to ChromaDB (persisted on disk)

── At Query Time ──

User Question → Embedded → Similarity Search → Top 3 Chunks

↓

Prompt Engineering → Context + Question sent to LLM

↓

Groq LLaMA 3 → Generates precise answer

↓

Streamlit UI → Displays answer + source page badges
```

---

## ⚙️ Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| UI | Streamlit | Chat interface |
| PDF Parsing | PyPDF | Extract text from PDFs |
| Chunking | LangChain RecursiveCharacterTextSplitter | Smart text splitting |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Local free embeddings |
| Vector DB | ChromaDB | Store and search vectors |
| LLM | Groq LLaMA 3.3 70B | Answer generation |
| Orchestration | LangChain LCEL | Pipeline management |
| Logging | Python logging | Professional log system |
| Config | python-dotenv | Secure API key management |

---

## 📁 Project Structure
```
DocuMind-AI/
│
├── src/
│   ├── document_loader.py   ← PDF reading and validation
│   ├── chunker.py           ← Recursive text chunking
│   ├── embedder.py          ← HuggingFace embedding model
│   ├── vector_store.py      ← ChromaDB save and load
│   ├── retriever.py         ← Similarity search
│   ├── qa_chain.py          ← LLM chain with prompt engineering
│   └── logger.py            ← Centralised logging system
│
├── data/
│   └── uploaded_docs/       ← PDF files stored here
│
├── vectorstore/             ← ChromaDB persisted vectors
├── logs/                    ← All module log files
├── app.py                   ← Streamlit application entry point
├── requirements.txt
├── .env                     ← API keys (never committed)
├── .gitignore
└── README.md
```

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/hemamalini0708/DocuMind-AI-PDF-Q-A-System-using-RAG-LLM.git
cd DocuMind-AI-PDF-Q-A-System-using-RAG-LLM
```

### 2. Create Virtual Environment
```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys
Create a `.env` file in the root folder:
```bash
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_token_here
```
Get your free Groq API key at: https://console.groq.com

### 5. Run the App
```bash
streamlit run app.py
```

Open in browser: `http://localhost:8501`

---

## 💡 How to Use

1. Open the app in your browser
2. Upload one or more PDF files from the sidebar
3. Click **🚀 Process Document(s)**
4. Wait for the ✅ success message
5. Type your question in the chat box
6. Get your answer with source page references instantly

---

## 🧪 Tested On

| Document | Pages | Chunks | Result |
|---|---|---|---|
| Attention Is All You Need (Research Paper) | 15 | 52 | ✅ Accurate |
| Fairness and Machine Learning (Textbook) | 39 | 141 | ✅ Accurate |
| Tesla Impact Report 2023 (Corporate Report) | 160 | 325 | ✅ Accurate |
| ML Intern Resume (Single Page) | 1 | 4 | ✅ Accurate |

---

## ✨ Key Features

- ✅ Multi-PDF support — upload and query multiple PDFs at once
- ✅ Source page citations — every answer shows which pages it came from
- ✅ Chat history — full conversation memory within session
- ✅ Clear & reload — switch documents anytime with one click
- ✅ Prompt engineering — LLM only answers from document, never hallucinates
- ✅ Professional logging — every module writes structured logs to file
- ✅ Fully free stack — Groq free tier + local HuggingFace embeddings

---

## 📊 RAG Pipeline Performance

| Document Size | Processing Time | Answer Time |
|---|---|---|
| 1 page | ~6 seconds | < 1 second |
| 15 pages | ~7 seconds | < 1 second |
| 39 pages | ~14 seconds | < 1 second |
| 160 pages | ~30 seconds | < 1 second |

---

## 🔑 Environment Variables

| Variable | Description | Where to Get |
|---|---|---|
| `GROQ_API_KEY` | Groq LLM API key | https://console.groq.com |
| `HUGGINGFACE_API_KEY` | HuggingFace token | https://huggingface.co/settings/tokens |

---

## 📦 Requirements
```
streamlit==1.43.2
langchain==1.2.13
langchain-community==0.4.1
langchain-groq==1.1.2
langchain-huggingface
langchain-chroma
chromadb==0.6.3
sentence-transformers==4.0.1
pymupdf==1.25.4
python-dotenv==1.1.0
```

---

## 👩‍💻 Author

**Gangumalla Hema Malini**
Aspiring AI/ML Engineer | GenAI & RAG Systems Enthusiast

📧 hemamalinig07@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/hemamalinig07)
🐙 [GitHub](https://github.com/hemamalini0708)

---

## 📄 License

MIT License — free to use and modify with attribution.
