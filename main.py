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

        # Load a more accurate Hugging Face model for question-answering
        @st.cache_resource
        def load_qa_pipeline():
            return pipeline("question-answering", model="deepset/roberta-base-squad2")

        qa_pipeline = load_qa_pipeline()

        # Input for questions
        question = st.text_input("Ask a question about the PDF content:")

        if question:
            # Get the answer from the model
            result = qa_pipeline(question=question, context=text)

            # Extract answer and display
            answer = result.get("answer", None)
            if answer and answer.strip():
                st.subheader("Answer:")
                st.write(answer)
            else:
                st.error("Sorry, I couldn't find a suitable answer. Try rephrasing the question.")
    else:
        st.error("No text could be extracted from the PDF. Please upload a valid PDF.")
else:
    st.info("Please upload a PDF to start.")
