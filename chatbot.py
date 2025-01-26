import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import torch
import streamlit as st


def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def split_text_into_chunks(text, chunk_size=500):
    """Split text into smaller chunks."""
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        current_length += len(sentence.split())
        current_chunk.append(sentence)
        if current_length > chunk_size:
            chunks.append('. '.join(current_chunk))
            current_chunk = []
            current_length = 0

    if current_chunk:
        chunks.append('. '.join(current_chunk))
    
    return chunks

def generate_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    """Generate embeddings for document chunks."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings

def create_faiss_index(embeddings):
    """Create a FAISS index from embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

def get_answer(question, context):
    """Answer a question using a Q&A model."""
    qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2", device=-1)
    response = qa_model(question=question, context=context)
    return response['answer']

def find_relevant_chunk(question, chunks, index, embeddings, model):
    """Find the most relevant chunk using cosine similarity."""
    question_embedding = model.encode(question, convert_to_tensor=True)
    question_embedding_np = question_embedding.detach().cpu().numpy()
    distances, indices = index.search(question_embedding_np.reshape(1, -1), k=1)
    return chunks[indices[0][0]]

def answer_question(question, chunks, index, embeddings, model):
    """Retrieve the relevant chunk and answer the question."""
    relevant_chunk = find_relevant_chunk(question, chunks, index, embeddings, model)
    answer = get_answer(question, relevant_chunk)
    return answer


def main():
    # Initialize the app
    st.title("Research Assistant Chatbot")

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

    # Process the document
    if uploaded_file is not None:
        st.write("Processing document...")
        document_text = extract_text_from_pdf(uploaded_file)
        chunks = split_text_into_chunks(document_text)
        st.write(f"Document processed into {len(chunks)} chunks.")

        # Generate embeddings
        model_name = "all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)
        embeddings = generate_embeddings(chunks, model_name)
        index = create_faiss_index(embeddings)

        # Ask a question
        question = st.text_input("Ask a question:")
        if question:
            answer = answer_question(question, chunks, index, embeddings, model)
            st.write(f"**Answer:** {answer}")
            
if __name__ == "__main__":
    main()
