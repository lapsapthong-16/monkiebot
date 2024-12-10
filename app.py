import streamlit as st
# Set page config first, before any other st commands
st.set_page_config(page_title="PDF QA System", layout="wide")

# fix tokenizer parallelism warning
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import fitz  # PyMuPDF
from groq import Groq
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from sentence_transformers import SentenceTransformer
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langchain.embeddings import HuggingFaceInstructEmbeddings
import nltk
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Groq client
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
if not client:
    client = input("Please enter the Groq API key: ")
    os.environ["GROQ_API_KEY"]

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into text
    processed_text = ' '.join(tokens)
    return processed_text

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    #text-normalization
    processed_text = preprocess_text(text)
    return processed_text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # Default to the first embedding model if not set
    selected_embedding = st.session_state.get("selected_embedding")
    
    # Debugging: Print the selected embedding model
    #st.write(f"Using embedding model: {selected_embedding}")
    
    # Load the correct embedding model
    if selected_embedding == "hkunlp/instructor-large":
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=selected_embedding, 
            model_kwargs={"device": "mps"}
        )
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=selected_embedding,
            model_kwargs={"device": "mps"}
        )
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore

def chat_with_gpt(question, pdf_context):
    try:
        selected_model = st.session_state.get("selected_model")
        if pdf_context:
            # Use existing vectorstore to find relevant context
            relevant_docs = st.session_state.vectorstore.similarity_search(question, k=2)
            # Limit context length to ~2000 characters
            relevant_context = "\n".join([doc.page_content for doc in relevant_docs])[:2000]
        else:
            relevant_context = "No PDF context provided."
            
        # Use Groq to generate the answer
        answer_response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an academic assistant specialized in summarizing, analyzing, and answering questions "
                        "based on research papers, journal articles, and academic documents. Provide detailed and accurate "
                        "responses, referencing the provided context when applicable. Maintain a formal tone suitable for "
                        "academic discussions."
                    )
                },
                {
                    "role": "user",
                    "content": f"Context: {relevant_context}\nQuestion: {question}"
                }
            ],
            model=selected_model, 
            max_tokens=1000,
        )
        return answer_response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

def extract_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Sidebar for PDF Uploads and Model Selection
with st.sidebar:
    st.title("📋 Menu")
    st.write("Configure your settings below:")
    
    # Spinner for choosing the embedding model
    embedding_options = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-MiniLM-L3-v2",
        "hkunlp/instructor-large"
        ]
    selected_embedding = st.selectbox("🔎 Embedding Model:", embedding_options)

    # Spinner for choosing the language model
    model_options = ["llama3-8b-8192", "gemma-7b-it"]
    selected_model = st.selectbox("🧠 Language Model:", model_options)

    pdf_docs = st.file_uploader(
        "📂 Upload PDF Files:", 
        accept_multiple_files=True, 
        type=["pdf"]  # Restricting file types to PDF
    )
    st.write("---")

    if st.button("🔄 Submit & Process"):
        if not pdf_docs:
            st.warning("⚠️ Please upload at least one PDF file!")
        else:
            with st.spinner("🚀 Processing PDF..."):
                try:
                    # Debugging: Print the selected embedding model
                    #st.write(f"Selected embedding model: {selected_embedding}")
                    
                    # Get and preprocess text
                    raw_text = get_pdf_text(pdf_docs)
                    # Create text chunks and vectorstore
                    text_chunks = get_text_chunks(raw_text)
                    
                    # Ensure selected_embedding is saved to session_state
                    if not selected_embedding:
                        st.session_state["selected_embedding"] = embedding_options[0]  # Set a default
                    else:
                        st.session_state["selected_embedding"] = selected_embedding
                    
                    st.session_state.vectorstore = get_vectorstore(text_chunks)
                    st.session_state["pdf_text"] = raw_text
                    st.session_state["selected_model"] = selected_model
                    st.success("PDF processed successfully!")
                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")

    # Clear Chat History Button
    if st.button("🗑️ Clear Chat History"):
        st.session_state["messages"] = []
        st.session_state["pdf_text"] = None
        st.session_state.vectorstore = None
        st.success("✅ Chat history cleared!")


# Default PDF context in session state
if "pdf_text" not in st.session_state:
    st.session_state["pdf_text"] = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Main content area
def main():
    st.title("Monkie Chatbot🐵")
    st.write("""
    Hi there! I'm **Monkie Chatbot**, your academic assistant. Let's explore academic papers togehter! ⸜(｡˃ ᵕ ˂)⸝♡
    """)

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "What can I help you with?"}
        ]

    # Display chat history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask something..."):
        # Add user message to chat history
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_with_gpt(prompt, st.session_state["pdf_text"])
                st.write(response)
                st.session_state["messages"].append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()