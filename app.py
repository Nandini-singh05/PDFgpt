import streamlit as st
from pypdf import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma  # Import Chroma instead of FAISS
import pickle
import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq

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

def load_embeddings():
    """Download and load SentenceTransformer embeddings only once per session."""
    if 'embeddings' not in st.session_state:
        with st.spinner("Downloading embeddings, please wait..."):
            st.session_state.embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.success("Embeddings loaded!")

def main():
    st.header("Chat with your PDF📄")

    # Load embeddings the first time the app runs
    load_embeddings()

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

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", 'rb') as f:
                vector_store = pickle.load(f)
        else:
            # Use the embeddings stored in session_state
            embeddings = st.session_state.embeddings
            
            # Use Chroma for vector storage instead of FAISS
            vector_store = Chroma.from_texts(chunks, embedding=embeddings)

            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_store, f)
        
        # Accept user question/query
        query = st.text_input("Ask questions about the PDF here:")
        
        if query:
            docs = vector_store.similarity_search(query=query, k=3)
            llm = ChatGroq(model="llama3-8b-8192")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(response)

if __name__ == "__main__":
    main()
