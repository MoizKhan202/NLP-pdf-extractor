import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader

# Title and description
st.title("PDF Question and Answering App")
st.write("Upload a PDF, and ask questions based on its content!")

# File uploader
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    # Extract text from the uploaded PDF
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    if text.strip():
        st.success("PDF text successfully extracted!")

        # Show extracted text in a collapsible section
        with st.expander("Extracted PDF Text"):
            st.write(text)

        # Question input
        question = st.text_input("Ask a question about the PDF content:")

        if question:
            # Use Hugging Face pipeline for Q&A
            qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

            # Get the answer from the model
            result = qa_pipeline(question=question, context=text)
            answer = result["answer"]

            # Display the answer
            st.subheader("Answer:")
            st.write(answer)
    else:
        st.error("No text could be extracted from the PDF. Please upload a valid PDF.")
else:
    st.info("Please upload a PDF to start.")

