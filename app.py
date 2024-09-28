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
import base64

# Sidebar contents
with st.sidebar:
    st.title("📄🤗 PDFgpt : Chat with your PDF")
    add_vertical_space(1)
    st.markdown('''### About PDFgpt:
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

def autoplay_audio(file_path: str):
    """Auto-play the audio in the Streamlit app."""
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true" style="width: 100%;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)

def main():
    st.header("Chat with your PDF📄")

    # Ask user to upload PDF
    pdf = st.file_uploader("Upload PDF here", type="pdf")
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)
        
        # Store name derived from the PDF file name
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}_chunks.pkl"):
            # Load stored chunks and recreate vector store
            with open(f"{store_name}_chunks.pkl", 'rb') as f:
                chunks = pickle.load(f)
            
            with st.spinner("Recreating the vector store..."):
                embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vector_store = Chroma.from_texts(chunks, embedding=embeddings)
        else:
            # Show progress message during embedding download
            with st.spinner("Downloading and loading embeddings, please wait..."):
                embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            st.success("Embeddings loaded successfully!")
            
            # Use Chroma for vector storage instead of FAISS
            vector_store = Chroma.from_texts(chunks, embedding=embeddings)

            # Store the chunks for future use
            with open(f"{store_name}_chunks.pkl", "wb") as f:
                pickle.dump(chunks, f)
        
        # Accept user question/query
        query = st.text_input("Ask questions about the PDF here:")
        
        if query:
            # Clear previous audio file from session state to ensure new audio plays
            if 'audio_played' in st.session_state:
                del st.session_state.audio_played

            docs = vector_store.similarity_search(query=query, k=3)
            llm = ChatGroq(model="llama3-8b-8192")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)

            # Generate a unique filename for each response to avoid caching
            speech_file = f"response_{query.replace(' ', '_')}.mp3"
            text_to_speech(response, speech_file)

            # Play the generated speech with autoplay
            autoplay_audio(speech_file)

            # Store a flag in session state to keep track of played audio
            st.session_state.audio_played = True

            # Display the response text below the audio player
            st.write(response)

if __name__ == "__main__":
    main()

