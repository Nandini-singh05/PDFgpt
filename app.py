import streamlit as st
from pypdf import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import pickle
import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from gtts import gTTS
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import fitz  # PyMuPDF
from io import BytesIO

# Sidebar contents
with st.sidebar:
    st.title("📄🤗 PDFgpt : Chat with your PDF")
    add_vertical_space(1)
    st.markdown('''
    ### About PDFgpt:
    This application is an LLM-powered chatbot built using the following:
    - [Langchain](https://www.langchain.com/)
    - [Streamlit](https://streamlit.io/)
    - [Hugging Face](https://huggingface.co/)
    - [Groq](https://groq.com/groqcloud/)
    ''')
    add_vertical_space(4)
    st.write('Made with ❤️ by [Nandini Singh](http://linkedin.com/in/nandini-singh-bb7154159)')

def text_to_speech(text, filename):
    """Convert text to speech and save as an MP3 file."""
    tts = gTTS(text)
    tts.save(filename)

def extract_text_from_image_pdf(pdf):
    """Extract text from an image-based PDF using OCR."""
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
    """Extract text from a standard text-based PDF."""
    if pdf is not None:
        text = ""
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
        return text

def main():
    st.header("Chat with your PDF📄")

    # Ask user to upload PDF
    pdf = st.file_uploader("Upload PDF here", type="pdf")
    
    if pdf is not None:
        # First attempt to extract text from the PDF (text-based PDF)
        text = extract_text_from_pdf(pdf)
        
        if not text:  # If no text is extracted, try OCR (image-based PDF)
            st.warning("No text found. Attempting OCR on image-based PDF.")
            text = extract_text_from_image_pdf(pdf)
        
        if text:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )

            chunks = text_splitter.split_text(text=text)

            # Check if chunks are not empty before proceeding
            if chunks:
                store_name = pdf.name[:-4]

                if os.path.exists(f"{store_name}_chunks.pkl"):
                    # Load stored chunks and recreate vector store
                    with open(f"{store_name}_chunks.pkl", 'rb') as f:
                        chunks = pickle.load(f)

                    with st.spinner("Recreating the vector store..."):
                        embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                        vector_store = Chroma.from_texts(chunks, embedding=embeddings)
                else:
                    with st.spinner("Downloading and loading embeddings, please wait..."):
                        embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

                    st.success("Embeddings loaded successfully!")
                    
                    vector_store = Chroma.from_texts(chunks, embedding=embeddings)

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
            else:
                st.error("No text found in PDF to create chunks. Please check your PDF content.")
        else:
            st.error("No text found or OCR failed to extract text. Please check the PDF.")
    
if __name__ == "__main__":
    main()
