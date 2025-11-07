# MonkieBot 🙉 — PDF Q&A with Retrieval + Groq LLM

MonkieBot is a Streamlit app that lets you upload one or more PDFs, builds a searchable vector index over their contents, and then answers your questions using a Groq-hosted LLM with retrieved context from the PDFs. It also includes quality metrics (ROUGE-1 and BERTScore) for a few sample Q&A, and can export the full chat transcript to PDF.

# ✨ Features

* **Multi-PDF ingestion:** Extracts text from PDFs (via PyMuPDF) with block sorting to handle multi-column layouts.
* **Text preprocessing:** lowercasing, punctuation/number removal, tokenization, stopword removal, and lemmatization (NLTK).
* **Chunking & indexing:** Splits text into overlapping chunks and indexes with **FAISS** using Hugging Face embeddings (or Instructor embeddings).
* **Retrieval-augmented generation (RAG):** Retrieves the most relevant chunks and sends them as context to a **Groq** LLM (Llama 3.3 70B / Gemma 2 9B / Mixtral 8x7B).
* **Conversation memory & UI:** Chat interface with persistent session state; clear history; PDF export of either the full chat or selected messages.
* **Evaluation metrics:** Optional ROUGE-1 and **BERTScore** to compare model outputs against a small set of ground-truth answers (for demo purposes).
* **Configurable models:** Choose embedding and LLM models from the sidebar.

# 🧱 Architecture (High Level)

1. **Ingest PDFs** → extract text with PyMuPDF (`fitz`) → preprocess (NLTK)
2. **Split** → `CharacterTextSplitter` creates ~1,000-char chunks with 200 overlap
3. **Embed** → HuggingFace embeddings (or Instructor) on CPU
4. **Index** → FAISS vectorstore in session
5. **Ask** → Retrieve top-k chunks → build prompt with context → Groq LLM
6. **Answer** → Display in Streamlit chat; (optional) score with ROUGE/BERTScore
7. **Export** → Save conversation to PDF via ReportLab

# 🚀 Quickstart

## 1) Prerequisites

* Python 3.10+ (3.11 recommended)
* pip (or uv/poetry)
* Groq API key (get one at groq.com)
* CPU is fine; GPU not required.

> Note: Windows users should install `faiss-cpu` (already in the requirements below).

## 2) Clone & install

```bash
git clone <your-repo-url>
cd <your-repo>
```

Create a virtual env (recommended):

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 3) Configure environment

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_api_key_here
```

The code also prompts for a key at runtime if the env var is missing.

## 4) NLTK data

The app downloads the required corpora at startup: **stopwords**, **punkt**, **wordnet**, and **punkt_tab** (newer NLTK versions).

If you’re running in a restricted environment, you can pre-download:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')  # present in NLTK >=3.9
nltk.download('wordnet')
```

## 5) Run the app

Save the provided code as `app.py`, then:

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

# 📁 Project Structure

```
.
├─ app.py
├─ requirements.txt
├─ .env                 # contains GROQ_API_KEY
└─ README.md
```

# 🙌 Acknowledgments

* Groq for blazing-fast LLM inference.
* LangChain, FAISS, PyMuPDF, NLTK, Sentence Transformers, ReportLab, Streamlit community.

# Run it now

```bash
pip install -r requirements.txt
echo "GROQ_API_KEY=your_api_key" > .env
streamlit run app.py
```
