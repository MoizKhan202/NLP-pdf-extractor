import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch

# Function to load and cache QA pipeline
@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="deepset/roberta-base-squad2")

# Function to preprocess and chunk text
def preprocess_text(text, chunk_size=500):
    sentences = text.split(". ")
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(". ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
    if current_chunk:
        chunks.append(". ".join(current_chunk))
    return chunks

# Function to retrieve the most relevant chunk
def retrieve_relevant_chunk(query, chunks, embedder):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)
    most_relevant_idx = torch.argmax(similarities).item()
    return chunks[most_relevant_idx]

# Streamlit UI
st.title("Fast and Accurate PDF Question Answering")
st.write("Upload a PDF, and ask questions about its content!")

uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    # Extract text from PDF
    pdf_reader = PdfReader(uploaded_file)
    text = " ".join(page.extract_text() for page in pdf_reader.pages)

    if text.strip():
        st.success("PDF text successfully extracted!")

        # Show extracted text in a collapsible section
        with st.expander("Extracted PDF Text"):
            st.write(text)

        # Load QA model and embedder
        qa_pipeline = load_qa_pipeline()
        embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")

        # Preprocess and chunk text
        chunks = preprocess_text(text)

        # Ask question
        question = st.text_input("Ask a question about the PDF content:")

        if question:
            # Retrieve relevant chunk
            relevant_chunk = retrieve_relevant_chunk(question, chunks, embedder)

            # Get answer from QA model
            result = qa_pipeline(question=question, context=relevant_chunk)
            answer = result.get("answer", "No suitable answer found.")

            st.subheader("Answer:")
            st.write(answer)
    else:
        st.error("No text could be extracted from the PDF. Please upload a valid PDF.")
else:
    st.info("Please upload a PDF to start.")
