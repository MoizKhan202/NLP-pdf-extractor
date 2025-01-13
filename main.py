import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader

# Initialize the question-answering model
@st.cache_resource
def load_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

qa_model = load_model()

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# Streamlit App
st.title("PDF Question Answering App")
st.markdown("""
**Instructions**:
1. Upload a PDF file.
2. Type your question in the input box.
3. Click "Get Answer" to retrieve an answer from the PDF content.
""")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Extract text from the uploaded PDF
    pdf_text = extract_text_from_pdf(uploaded_file)
    if pdf_text:
        st.success("PDF successfully uploaded and text extracted!")
        
        # Optionally display a snippet of the text
        st.text_area("Extracted Text (Snippet)", pdf_text[:1000] + "...", height=200)

        # Question input
        question = st.text_input("Enter your question:")
        
        if st.button("Get Answer"):
            if question.strip() and pdf_text.strip():
                try:
                    # Get answer from the model
                    response = qa_model(question=question, context=pdf_text)
                    st.success("Answer: " + response['answer'])
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("Please upload a PDF and enter a valid question.")
    else:
        st.error("Could not extract text from the uploaded PDF.")
else:
    st.info("Please upload a PDF to get started.")
