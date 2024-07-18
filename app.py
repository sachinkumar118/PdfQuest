import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key environment variable
openai_key = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = openai_key

# Function to read text from PDF
def read_pdf(pdf_file):
    raw_text = ''
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        raw_text += page.extract_text()
    return raw_text

# Function to split text into manageable chunks
def split_text(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(raw_text)

# Initialize Streamlit app
def main():
    st.set_page_config(page_title="PdfQuest")

    st.header("PdfQuest") 
    # File upload and processing
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        st.text('Processing PDF...')

        # Read and split text
        raw_text = read_pdf(uploaded_file)
        texts = split_text(raw_text)

        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings()

        # Create FAISS vector store from texts
        document_search = FAISS.from_texts(texts, embeddings)

        # Load Question Answering chain
        chain = load_qa_chain(OpenAI(), chain_type="stuff")

        # User query input
        query = st.text_input('Enter your question:')
        if st.button('Ask'):
            if query:
                st.text('Searching for answer...')
                docs = document_search.similarity_search(query)
                result = chain.run(input_documents=docs, question=query)
                st.write('Answer:', result)

if __name__ == "__main__":
    main()
