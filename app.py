import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from multiprocessing import Pool
import pickle
import hashlib

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            return ''.join(page.extract_text() or '' for page in pdf_reader.pages)
    except Exception as e:
        st.error(f"Error processing '{file_path}': {str(e)}")
        return ""

@st.cache_data
def get_pdf_text_from_resources():
    resources_folder = "Resources"
    if not os.path.exists(resources_folder):
        st.error(f"The '{resources_folder}' directory does not exist.")
        return None

    pdf_files = [os.path.join(resources_folder, f) for f in os.listdir(resources_folder) if f.lower().endswith('.pdf')]
    if not pdf_files:
        st.warning(f"No PDF files found in the '{resources_folder}' directory.")
        return None

    # Calculate hash of PDF files to use as cache key
    pdf_hash = hashlib.md5(''.join(pdf_files).encode()).hexdigest()
    cache_file = f"pdf_cache_{pdf_hash}.pkl"

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    combined_text = ""
    progress_bar = st.progress(0)
    for i, pdf_file in enumerate(pdf_files):
        text = extract_text_from_pdf(pdf_file)
        combined_text += text
        progress = (i + 1) / len(pdf_files)
        progress_bar.progress(progress)

    if not combined_text:
        st.warning("No text could be extracted from any of the PDF files.")
        return None

    with open(cache_file, 'wb') as f:
        pickle.dump(combined_text, f)

    return combined_text

@st.cache_data
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def get_vector_store(text_chunks):
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

@st.cache_resource
def get_conversational_chain():
    prompt_template = """
    Context:\n {context}?\n
    Question: \n{question}\n
    Audience: \n{audience}\n

    Please provide an answer suitable for the specified audience.

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "audience"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, audience, vector_store):
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question, "audience": audience},
        return_only_outputs=True
    )
    return response["output_text"]

def format_response(response, audience):
    if audience == "Space Science for Kids":
        return f"🚀✨ {response} \n\nWant to know more? Ask another question!"
    elif audience == "Space Science for Student":
        return f"🔭📚 {response} \n\nReferences available upon request."
    elif audience == "Space Science for Professional":
        return f"📊🌌 {response} \n\nPeer-reviewed sources available in citations."
    return response

def main():
    st.set_page_config("Space Science Chatbot")
    st.title("🌌 Space Science Chatbot")
    st.subheader("Explore the cosmos at your level!")

    audience = st.selectbox(
        "Select your audience type:",
        ("Space Science for Kids", "Space Science for Student",
         "Space Science for Professional")
    )

    st.write("Starting PDF processing...")
    with st.spinner("Processing PDFs from Resources folder..."):
        raw_text = get_pdf_text_from_resources()
        if raw_text is None:
            st.error("Unable to process PDFs. Please check the warnings and errors above.")
            return

        st.write("PDF processing completed. Creating text chunks...")
        text_chunks = get_text_chunks(raw_text)
        st.write(f"Number of text chunks created: {len(text_chunks)}")

        st.write("Creating vector store...")
        vector_store = get_vector_store(text_chunks)

    st.success("PDFs processed successfully!")

    user_question = st.text_input("Ask a question about space science:")

    if user_question:
        with st.spinner("Generating response..."):
            response = user_input(user_question, audience, vector_store)
            formatted_response = format_response(response, audience)
            st.write("Reply:", formatted_response)

if __name__ == "__main__":
    main()