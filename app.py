import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import pickle
import os
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
# from langchain.prompts import ChatPromptTemplate
from langchain.chains.question_answering import load_qa_chain

from secret_key import GROQ_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# sidebar contents here:
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


def main():
    st.header("Chat with your PDF📄")

    # Ask user to upload pdf 

    pdf = st.file_uploader("Upload PDF here", type="pdf")
    # st.write(pdf.name)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        # st.write(pdf_reader)
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )

        chunks = text_splitter.split_text(text=text)
        
        # embeddings - using hugging face

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", 'rb') as f:
                vector_store = pickle.load(f)
            # st.write('embeddings loaded from the disk.')
        else:
            
            embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 

            vector_store = FAISS.from_texts(chunks, embedding=embeddings)

            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_store, f)
            
            # st.write('Embeddings written to the disk.')
        # print("done embeddings")

        # Accept user question/query
        query = st.text_input("Ask questions about the PDF here:")
        st.write(query)

        if query:
            docs = vector_store.similarity_search(query=query, k=3)
            llm = ChatGroq(model="llama3-8b-8192")
            chain = load_qa_chain(llm = llm, chain_type="stuff")
            response = chain.run(input_documents = docs, question=query)
            st.write(response)
            # st.write(docs)

        # st.write(chunks)
        # st.write(text)

if __name__ == "__main__":
    main()