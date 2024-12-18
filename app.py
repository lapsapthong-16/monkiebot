
import os
import time
import string
import re
from io import BytesIO
os.environ["TOKENIZERS_PARALLELISM"] = "false" # fix tokenizer parallelism warning
import streamlit as st
import fitz 
from groq import Groq
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer 
from bert_score import score as bert_score
from reportlab.pdfgen import canvas
from dotenv import load_dotenv
#from langchain.llms import HuggingFaceHub


# Load environment variables
load_dotenv()
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Initialize the Groq client
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
if not client:
    client = input("Please enter the Groq API key: ")
    os.environ["GROQ_API_KEY"]

def initialize_session_state():
    if "pdf_text" not in st.session_state:
        st.session_state["pdf_text"] = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # Start with an empty chat history

initialize_session_state()

def clear_chat_history():
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Submit your PDFs and ask a question!"}
    ]
    st.session_state["pdf_text"] = None
    st.session_state.vectorstore = None
    st.success("✅ Chat history cleared!")

def create_pdf(conversation):
    # Initialize a byte stream to hold the PDF
    pdf_buffer = BytesIO()
    
    # Create a canvas for the PDF
    c = canvas.Canvas(pdf_buffer)
    c.setTitle("Conversation Transcript")
    
    # Add conversation content to the PDF
    y_position = 800  # Starting vertical position
    line_height = 14  # Space between lines
    
    for msg in conversation:
        role = msg.get("role", "User").capitalize()
        content = msg.get("content", "")
        
        line = f"{role}: {content}"
        
        if y_position <= 50:  # Add a new page if space runs out
            c.showPage()
            y_position = 800
            
        c.drawString(50, y_position, line)
        y_position -= line_height
    
    # Save the PDF to the byte stream
    c.save()
    pdf_buffer.seek(0)  # Reset the buffer position to the start
    
    return pdf_buffer.getvalue()  # Return the PDF content as bytes

def handle_pdf_export():
    # Ensure that the messages session state is initialized and has content
    if "messages" not in st.session_state or not st.session_state["messages"]:
        st.warning("⚠️ Start a conversation first to enable PDF export!")
        return

    # Check if there is at least one assistant response
    if not any(msg.get("role") == "assistant" for msg in st.session_state["messages"]):
        st.warning("⚠️ Wait for the assistant to reply before exporting the conversation!")
        return

    # Create PDF with the full conversation
    pdf_bytes = create_pdf(st.session_state["messages"])
    st.download_button(
        label="💾 Download Conversation as PDF",
        data=pdf_bytes,
        file_name="conversation.pdf",
        mime="application/pdf",
    )

    # Multiselect for selecting specific messages to export
    selected_messages = st.multiselect(
        "Select messages to download:",
        options=[f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state["messages"]],
    )

    if selected_messages:
        selected_conversation = [
            msg for msg in st.session_state["messages"]
            if f"{msg['role'].capitalize()}: {msg['content']}" in selected_messages
        ]
        selected_pdf_bytes = create_pdf(selected_conversation)
        st.download_button(
            label="💾 Download Selected Conversation as PDF",
            data=selected_pdf_bytes,
            file_name="selected_conversation.pdf",
            mime="application/pdf",
        )
    else:
        st.warning("⚠️ Please select at least one message to download!")

def setup_ui():
    st.set_page_config(page_title="PDF QA System", layout="wide")

    with st.sidebar:
        st.title("📋 Menu")
        st.write("Configure your settings below:")

        embedding_options = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "hkunlp/instructor-large"
        ]

        # Extract only the part after the last "/"
        display_embedding_options = [option.split("/")[-1] for option in embedding_options]

        # Create a mapping to store the full names against the display names
        embedding_mapping = {option.split("/")[-1]: option for option in embedding_options}

        # Display the shortened names in the dropdown
        selected_embedding_display = st.selectbox("🔎 Embedding Model:", display_embedding_options)

        # Get the full name of the selected embedding
        selected_embedding = embedding_mapping[selected_embedding_display]

        # Store the selected embedding in session state
        st.session_state["selected_embedding"] = selected_embedding

        # Language Model Selection
        model_options = ["llama-3.3-70b-versatile", "gemma-7b-it", "mixtral-8x7b-32768"]
        selected_model = st.selectbox("🧠 Language Model:", model_options)
        st.session_state["selected_model"] = selected_model

        # PDF Upload
        pdf_docs = st.file_uploader(
            "📂 Upload PDF Files:", 
            accept_multiple_files=True, 
            type=["pdf"]
        )

        if st.button("🔄 Submit & Process"):
            process_pdfs(pdf_docs)

        if st.button("🗑️ Clear Chat History"):
            clear_chat_history()

        st.write("---")

        # PDF Export Options
        st.subheader("📤 Download & Share")
        handle_pdf_export()

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
        # Open the PDF file with PyMuPDF
        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        for page in doc:
            # Extract text using PyMuPDF's block-based text extraction
            blocks = page.get_text("blocks")
            # Sort blocks by vertical position (y0) to handle multi-column PDFs
            sorted_blocks = sorted(blocks, key=lambda b: b[1])
            for block in sorted_blocks:
                text += block[4] + "\n"  # Add text content from each block
    # Normalize text using the preprocess function
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
    
    # Load the correct embedding model
    if selected_embedding == "hkunlp/instructor-large":
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=selected_embedding, 
            model_kwargs={"device": "cpu"}
        )
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=selected_embedding,
            model_kwargs={"device": "cpu"}
        )
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore

def process_pdfs(pdf_docs):
    if not pdf_docs:
        st.warning("⚠️ Please upload at least one PDF file!")
        return
    
    with st.spinner("🚀 Processing PDF..."):
        try:
            # Step 1: Extract text from PDFs
            raw_text = get_pdf_text(pdf_docs)

            # Step 2: Split text into chunks
            text_chunks = get_text_chunks(raw_text)

            # Step 3: Create vectorstore
            st.session_state["vectorstore"] = get_vectorstore(text_chunks)

            # Save raw text in session state
            st.session_state["pdf_text"] = raw_text
            st.success("PDF processed successfully!")
        except Exception as e:
            st.error(f"❌ An error occurred during processing: {e}")

def generate_response(question):
    try:
        question = preprocess_text(question)
        vectorstore = st.session_state.get("vectorstore")
        context = ""

        # If vectorstore exists, fetch relevant documents for context
        if vectorstore:
            relevant_docs = vectorstore.similarity_search(question, k=2)
            context = "\n".join([doc.page_content for doc in relevant_docs])[:2000]

        # Measure the response time
        start_time = time.time()  # Start the timer

        # Send request to language model
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Academic assistant system prompt here."},
                {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
            ],
            model=st.session_state.get("selected_model"),
            max_tokens=1000
        )

        end_time = time.time()  # End the timer
        response_time = end_time - start_time  # Calculate response time

        # Extract the response content
        response_content = response.choices[0].message.content

        # Return both the response and the time taken
        return response_content, response_time

    except Exception as e:
        return f"An error occurred: {e}", None

def compute_bertscore(reference, candidate):
    # Calculate BERTScore
    _, _, F1 = bert_score([candidate], [reference], lang='en', verbose=True)
    return F1.mean().item()  # Return only the F1 score

def chat_with_gpt(question, pdf_context):
    try:
        selected_model = st.session_state.get("selected_model")
        question = preprocess_text(question)

        # Fetch relevant context from vectorstore if available
        if pdf_context:
            relevant_docs = st.session_state.vectorstore.similarity_search(question, k=2)
            relevant_context = "\n".join([doc.page_content for doc in relevant_docs])[:2000]
        else:
            relevant_context = "No PDF context provided."

        # Debugging output
        print("Selected Model:", selected_model)
        print("Relevant Context:", relevant_context)

        # Start measuring response time
        start_time = time.time()

        # Generate the chatbot response
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

        # End measuring response time
        end_time = time.time()
        response_time = end_time - start_time

        response = answer_response.choices[0].message.content

        # Define ground truth answers
        ground_truth = {
            "Who are the authors?": "The authors of the paper are: Luca Braghieri, Ro'ee Levy, and Alexey Makarin",
            "What research design use in this research paper?": "Quasi-experimental estimates of the impact of social media on mental health by leveraging a unique natural experiment: the staggered introduction of Facebook across US colleges. Our analysis couples data on student mental health around the years of Facebook’s expansion with a generalized difference-in-differences empirical strategy.",
            "What are the two main datasets in analysis?": "The two main datasets used in the analysis provided in the context are the National College Health Assessment (NCHA) dataset and the Facebook expansion date dataset"
        }

        # Capture the chatbot's response
        chatbot_response = response.strip()

        # Get the expected answer from ground truth based on the question
        expected_answer = ground_truth.get(question, "")

        # Initialize ROUGE scores and BERT F1 Score
        rouge1_score = 0.0
        bert_f1 = 0.0

        # Check if expected_answer is empty
        if expected_answer:
            # Normalize both responses
            normalized_chatbot_response = preprocess_text(chatbot_response)
            normalized_expected_answer = preprocess_text(expected_answer)

            # Initialize ROUGE scorer
            scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
            scores = scorer.score(normalized_expected_answer, normalized_chatbot_response)

            # Extract ROUGE-1 score
            rouge1_score = scores['rouge1'].fmeasure

            # Compute BERT F1 Score
            bert_f1 = compute_bertscore(normalized_expected_answer, normalized_chatbot_response)

            # Debugging output
            print("ROUGE-1 Score:", rouge1_score)
            print("BERT F1 Score:", bert_f1)
        else:
            print("No expected answer found for the question.")

        # Return the response with metrics and response time
        response_with_metrics = (
            f"{chatbot_response}\n\n"
            f"ROUGE-1 Score: {rouge1_score:.4f}\n"
            f"BERT F1 Score: {bert_f1:.4f}\n"
            f"Response Time: {response_time:.2f} seconds"
        )
        return response_with_metrics
    except Exception as e:
        print("Error occurred:", e)  # Print the error for debugging
        return f"An error occurred: {e}"

# Main content area
def main():
    setup_ui()  # Setup sidebar and configurations

    st.title("MonkieBot 🙉")
    # Display the static header
    st.subheader("Submit your PDFs and ask a question!")  

    # Display chat history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask something..."):
        # Append the user's input to session state
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Generate the assistant's reply and append it to session state
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_with_gpt(prompt, pdf_context=True)  # Include context flag
                st.write(response)
                st.session_state["messages"].append({"role": "assistant", "content": response})

        # Force immediate synchronization of session state
        st.rerun()

if __name__ == "__main__":
    main()