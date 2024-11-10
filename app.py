import fitz  # PyMuPDF
import pytesseract
import io
from PIL import Image
from io import BytesIO
import streamlit as st
from pypdf2 import PdfReader
import re
import pickle
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from langchain_groq import ChatGroq
from gtts import gTTS
from chromadb import Client
import torch
from transformers import AutoTokenizer, AutoModel

# Initialize Hugging Face tokenizer and model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy()

client = Client()

def clean_collection_name(name):
    cleaned_name = re.sub(r'[^a-zA-Z0-9-_]', '-', name)
    cleaned_name = cleaned_name[:63].strip('-_')
    if len(cleaned_name) < 3:
        cleaned_name += 'pdf'
    return cleaned_name

def extract_text_from_image_pdf(pdf):
    try:
        pdf_document = fitz.open(stream=pdf.read())
        extracted_text = ""
        for page_num in range(pdf_document.page_count):
            try:
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap()
                img = Image.open(BytesIO(pix.tobytes()))
                ocr_text = pytesseract.image_to_string(img)
                extracted_text += ocr_text
            except Exception as e:
                st.warning(f"Error processing page {page_num}: {str(e)}")
        return extracted_text
    except Exception as e:
        st.error(f"Error opening the PDF: {str(e)}")
        return ""

def extract_text_from_pdf(pdf):
    if pdf is not None:
        text = ""
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
        return text

def process_pdf_for_embeddings(pdf):
    if pdf is None:
        return None
    
    pdf_type = st.radio(
        "Select the type of the PDF",
        ("Text-Based PDF", "Image-Based PDF (Requires OCR)")
    )

    if pdf_type == "Text-Based PDF":
        text = extract_text_from_pdf(pdf)
    elif pdf_type == "Image-Based PDF (Requires OCR)":
        text = extract_text_from_image_pdf(pdf)

    if not text:
        st.error("No text extracted from the PDF.")
        return None
    
    chunks = text.split("\n")
    
    if not chunks:
        st.error("No valid chunks found in the PDF text.")
        return None

    embeddings = get_embeddings(chunks)
    
    store_name = clean_collection_name(pdf.name[:-4])
    st.write(f"Embedding store name: {store_name}")
    
    # Check if embeddings data already exists
    if os.path.exists(f"{store_name}_data.pkl"):
        with open(f"{store_name}_data.pkl", "rb") as f:
            data = pickle.load(f)
            chunks = data["chunks"]
            embeddings = data["embeddings"]
        st.write('Loaded embeddings from disk.')
    else:
        data = {"chunks": chunks, "embeddings": embeddings}
        with open(f"{store_name}_data.pkl", "wb") as f:
            pickle.dump(data, f)
        st.write('Saved embeddings to disk.')

    # Create or get Chroma collection
    try:
        collection = client.get_collection(store_name)
    except Exception:
        collection = client.create_collection(name=store_name)
    
    # Add embeddings to the collection
    for idx, chunk in enumerate(chunks):
        collection.add(
            ids=[f"{store_name}_{idx}"],
            documents=[chunk],
            metadatas=[{"source": f"page_{idx+1}"}],
            embeddings=embeddings[idx].tolist(),
        )

    st.write(f"Done processing the PDF for embeddings. Total chunks: {len(chunks)}")

    return collection  # Return the collection for querying

def text_to_speech(text):
    """Convert text to speech and return as a BytesIO stream."""
    tts = gTTS(text)
    mp3_stream = BytesIO()
    tts.write_to_fp(mp3_stream)  # Use write_to_fp instead of save()
    mp3_stream.seek(0)  # Reset the stream position to the beginning
    return mp3_stream


# Assuming the required imports and initializations are already done as per your provided code

def main():
    st.title("PDF Text Extraction and Embedding with Hugging Face & Chroma")
    pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf:
        collection = process_pdf_for_embeddings(pdf)  # Retrieve the collection
    
    # Accept user question/query
    query = st.text_input("Ask questions about the PDF here:")
    
    # In the main function, where we call similarity_search:
    if query and collection:
        # Perform similarity search using Chroma's similarity_search method
        results = collection.similarity_search(query=query, k=5)

        # Wrap documents in the correct format
        docs = [Document(page_content=doc) for doc in results]

        # Initialize the LLM and QA chain
        llm = ChatGroq(model="llama3-8b-8192")
        chain = load_qa_chain(llm=llm, chain_type="stuff")

        # Run the QA chain
        response = chain.run(input_documents=docs, question=query)

        # Convert the response to speech
        mp3_stream = text_to_speech(response)

        # Play the audio directly from the memory stream
        st.audio(mp3_stream, format="audio/mp3")

        # Display the response text below the audio player
        st.write(response)

    elif not query:
        st.error("No text found in PDF to create chunks. Please check your PDF content.")

if __name__ == "__main__":
    main()
