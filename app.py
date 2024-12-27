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
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer 
from bert_score import score as bert_score
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
# nltk.data.path.append('/path/to/nltk_data')  # Update with your path

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
    if st.session_state["pdf_text"] is None:
        # Generate a response from the language model indicating no PDFs have been uploaded
        response = generate_response("No PDFs have been uploaded yet. Please upload a PDF to start the conversation.")
        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.success("✅ Chat history cleared! Please upload a PDF to continue.")
    else:
        st.session_state["messages"] = []
        st.session_state["pdf_text"] = None
        st.session_state.vectorstore = None
        st.success("✅ Chat history cleared!")

def create_pdf(conversation):
    # Initialize a byte stream to hold the PDF
    pdf_buffer = BytesIO()
    
    # Set up document and styles
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=50,
        bottomMargin=50,
    )
    styles = getSampleStyleSheet()
    content = []

    # Title for the document
    title = Paragraph("<b>MonkieBot Conversation Transcript</b>", styles['Title'])
    content.append(title)
    content.append(Spacer(1, 12))

    # Loop through the conversation to add messages
    for msg in conversation:
        if isinstance(msg, dict):  # Ensure msg is a dictionary
            role = msg.get("role", "User").capitalize()
            content_text = msg.get("content", "").strip()  # This should now work correctly
            
            # Format the text as "Role: Content"
            formatted_text = f"<b>{role}:</b> {content_text}"

            # Add to the PDF content with a clean style
            paragraph = Paragraph(formatted_text, styles['BodyText'])
            content.append(paragraph)
            content.append(Spacer(1, 12))  # Add spacing between entries
        else:
            print("Warning: Expected a dictionary but got:", msg)  # Debugging line

    # Add the model used at the end of the PDF
    selected_model = st.session_state.get("selected_model", "Unknown Model").split('/')[-1]

    # Build the PDF
    try:
        doc.build(content)
    except Exception as e:
        print(f"Error occurred while generating PDF: {e}")

    # Return the PDF as bytes
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

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
        # Create a list to hold the selected conversation messages
        selected_conversation = []
        
        # Populate the selected_conversation list based on user selection
        for selected in selected_messages:
            role, content = selected.split(": ", 1)  # Split role and content
            # Find the corresponding message in the session state
            for msg in st.session_state["messages"]:
                if msg['role'].capitalize() == role and msg['content'] == content:
                    selected_conversation.append(msg)
                    break  # Break to avoid duplicates

        # Create PDF for the selected conversation
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
        model_options = ["llama-3.3-70b-versatile", "gemma2-9b-it", "mixtral-8x7b-32768"]
        selected_model = st.selectbox("🧠 Language Model:", model_options)

        # Store the selected model in session state
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
        vectorstore = st.session_state.get("vectorstore")
        context = ""

        # Check if vectorstore exists
        if vectorstore is None:
            return "❌ Vectorstore is not initialized. Please upload a PDF first."

        # If vectorstore exists, fetch relevant documents for context
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

        # Get the model name used
        model_name = st.session_state.get("selected_model", "Unknown Model").split('/')[-1]

        # Format the assistant's response
        assistant_response = f"Assistant({model_name}): {response_content}"

        # Append the response to the session state
        st.session_state["messages"].append({"role": "assistant", "content": assistant_response})

        # Return the formatted response
        return assistant_response  # Ensure this is a string

    except Exception as e:
        return f"An error occurred: {e}"  # Ensure this is a string

def compute_bertscore(reference, candidate):
    # Calculate BERTScore
    _, _, F1 = bert_score([candidate], [reference], lang='en', verbose=True)
    return F1.mean().item()  # Return only the F1 score

def chat_with_gpt(question, pdf_context):
    try:
        selected_model = st.session_state.get("selected_model")
        if pdf_context:
            # Check if vectorstore exists
            if st.session_state.get("vectorstore") is None:
                return "❌ Vectorstore is not initialized. Please upload a PDF first.", None
            
            # Use existing vectorstore to find relevant context
            relevant_docs = st.session_state.vectorstore.similarity_search(question, k=2)
            # Limit context length to ~2000 characters
            relevant_context = "\n".join([doc.page_content for doc in relevant_docs])[:2000]
        else:
            relevant_context = "No PDF context provided."

        # Measure response time
        start_time = time.time()  # Start the timer

        # Preprocess the question for chatbot input
        processed_question = preprocess_text(question)

        # Initialize model based on user selection
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
                    "content": f"Context: {relevant_context}\nQuestion: {processed_question}"
                }
            ],
            model=selected_model, 
            max_tokens=1000,
        )
        
        end_time = time.time()  # End the timer
        response_time = end_time - start_time  # Calculate response time

        response = answer_response.choices[0].message.content

        # Define ground truth answers
        ground_truth = {
            "Who are the authors?": "The authors are Tara Khursheed, Mohd Yunus Khalil Ansari, and Danish Shahab.",
            "What was the seedling height for the control group after 30 days?": "The seedling height for the control group after 30 days was 10.5 inches.",
            "Why might higher concentrations of caffeine have an inhibitory effect on growth and yield?": "Higher concentrations of caffeine may have an inhibitory effect on growth and yield due to toxic effects such as uneven damage to meristematic cells, structural changes in chromosome constitution, reduced nutrition contents, or disturbances in the mechanism of assimilation.?",
            "Why is silver conductive ink significant in the electronics industry?": "Silver conductive ink is significant in the electronics industry due to its high electrical and thermal conductivity, low bulk resistivity, and ability to cure at low temperatures (473–573 K). It offers advantages in creating smooth, conductive tracks and exhibits better physical and electrical performance compared to other materials like copper.",
            "What was the relationship between temperature and the hardness of silver?": "As temperature increases, the hardness of silver decreases. This reduction is due to the disappearance of grain boundaries during particle diffusion, which allows larger particles to form, leading to reduced mechanical strength.",
            "What conclusions were drawn about the correlation between temperature and electrical performance?": "The study concluded that electrical conductivity increases with temperature. This is attributed to the diffusion of silver particles, which reduces voids and allows a denser, smoother structure to form, thereby facilitating electron transportation and reducing resistance￼."
        }

        # Capture the chatbot's response
        chatbot_response = response.strip()

        # Use the original question for ground truth lookup
        expected_answer = ground_truth.get(question, "")

        # Initialize ROUGE scores and BERT F1 Score
        rouge1_score = 0.0
        bert_f1 = 0.0

        # Check if expected_answer is empty
        if expected_answer:
            # Normalize both responses for scoring
            normalized_chatbot_response = preprocess_text(chatbot_response)
            normalized_expected_answer = preprocess_text(expected_answer)

            # Initialize ROUGE scorer
            scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
            scores = scorer.score(normalized_expected_answer, normalized_chatbot_response)

            # Extract ROUGE-1 score
            rouge1_score = scores['rouge1'].fmeasure

            # Initialize BERT F1 Score
            bert_f1 = compute_bertscore(normalized_expected_answer, normalized_chatbot_response)

            # Print scores to terminal
            print("User's Question:", question)
            print("ROUGE-1 Score:", rouge1_score)
            print("BERT F1 Score:", bert_f1)
        else:
            print("No expected answer found for the question.")

        # Return the response with model name and response time
        response_with_metrics = (
            f"{chatbot_response}\n\n"
            f"Response Time: {response_time:.2f} seconds\n"
            f"| Model Used: {selected_model.split('/')[-1]}"  # Display the model name
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
                #prompt = preprocess_text(prompt)
                response = chat_with_gpt(prompt, pdf_context=True)  # Include context flag
                st.write(response)
                st.session_state["messages"].append({"role": "assistant", "content": response})

        # Force immediate synchronization of session state
        st.rerun()

if __name__ == "__main__":
    main()
