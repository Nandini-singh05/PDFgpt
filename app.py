import streamlit as st
from pypdf import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  # Import Chroma instead of FAISS
import pickle
import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from gtts import gTTS
from PIL import Image
import pytesseract
import io
from uuid import uuid4

def main():
    st.header("Chat with your PDF📄")

    # Ask user to upload PDF
    pdf = st.file_uploader("Upload PDF here", type="pdf")
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""

        # Extract text from each page in the PDF
        for page in pdf_reader.pages:
            page_text = extract_text_from_page(page)
            if page_text:
                text += page_text

        # Check if any text was extracted
        if not text.strip():
            st.error("No text could be extracted from this PDF. It may be a scanned document without recognizable text.")
            return

        # Split the extracted text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Ensure there are chunks to process
        if not chunks:
            st.error("Text extraction or splitting failed, resulting in no chunks to process.")
            return

        # Store name derived from the PDF file name
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}_chunks.pkl"):
            # Load stored chunks and recreate vector store
            with open(f"{store_name}_chunks.pkl", 'rb') as f:
                chunks = pickle.load(f)
            
            with st.spinner("Recreating the vector store..."):
                embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vector_store = Chroma.from_texts(chunks, embedding=embeddings, ids=[str(uuid4()) for _ in chunks])
        else:
            # Show progress message during embedding download
            with st.spinner("Downloading and loading embeddings, please wait..."):
                embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            st.success("Embeddings loaded successfully!")
            
            # Use Chroma for vector storage, ensuring unique IDs for each chunk
            vector_store = Chroma.from_texts(chunks, embedding=embeddings, ids=[str(uuid4()) for _ in chunks])

            # Store the chunks for future use
            with open(f"{store_name}_chunks.pkl", "wb") as f:
                pickle.dump(chunks, f)
        
        # Accept user question/query
        query = st.text_input("Ask questions about the PDF here:")
        
        if query:
            docs = vector_store.similarity_search(query=query, k=3)
            llm = ChatGroq(model="llama3-8b-8192")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)

            # Generate a unique filename for each response to avoid caching
            speech_file = f"response_{query.replace(' ', '_')}.mp3"
            text_to_speech(response, speech_file)

            # Clear the previous audio player and play the new audio
            st.audio(speech_file, format="audio/mp3")

            # Display the response text below the audio player
            st.write(response)

if __name__ == "__main__":
    main()

