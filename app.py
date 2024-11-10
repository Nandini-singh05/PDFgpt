from uuid import uuid4

# Code inside the main function
def main():
    st.header("Chat with your PDF📄")

    # Ask user to upload PDF
    pdf = st.file_uploader("Upload PDF here", type="pdf")
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""

        for page in pdf_reader.pages:
            text += extract_text_from_page(page)
        
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
                vector_store = Chroma.from_texts(chunks, embedding=embeddings, ids=[str(uuid4()) for _ in chunks])
        else:
            # Show progress message during embedding download
            with st.spinner("Downloading and loading embeddings, please wait..."):
                embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            st.success("Embeddings loaded successfully!")
            
            # Use Chroma for vector storage instead of FAISS, add unique IDs for each chunk
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

