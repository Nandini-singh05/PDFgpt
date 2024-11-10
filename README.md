<h1> <p align="center"> PDFgpt 📄🤖 </p> </h1>

PDFgpt is an LLM-powered application that enables users to upload PDF documents and receive precise, context-based answers through both text and audio outputs. By leveraging advanced language models, it quickly analyzes document content, providing multimodal responses for a seamless and interactive experience. Whether it’s for research, manuals, or large documents, PDFgpt makes querying information fast, intuitive, and accessible.

<p align="center">
<img src="PDFgpt.png" width="800" height="400"/>
</p>
  
🌟 Try it here: [pdfgpt-llama-huggingface.streamlit.app](https://pdfgpt-llama-huggingface.streamlit.app/)


### ✨ Key Features
- 📁 **PDF Upload**: Users can upload PDFs up to **200MB** in size.
- 📝 **Multimodal Responses**: Receive answers in both **text** and **audio** (via Google Text-to-Speech).
- ⚡ **Advanced Language Models**: Powered by **LLaMA3-8b-8192** for **accurate**, **contextual** answers.
- 📜 **Text and Image-Based PDFs**: Supports both **text-based** and **image-based** PDFs (via OCR).
- 🧬 **Embedding & Search**: Utilizes **SentenceTransformers** for embeddings and **Chroma** for similarity search.

### 🛠️ Built With
- **[Langchain](https://www.langchain.com/):** 🧠 For handling text processing and interactions with language models.
- **[Streamlit](https://streamlit.io/):** 🖥️ For building the user interface.
- **[Hugging Face](https://huggingface.co/):** 🤗 For language models and embeddings.
- **[Groq](https://groq.com/groqcloud/):** ⚙️ For high-performance language model integration.
- **[Chroma](https://www.trychroma.com/):** 🔍 For storing and searching embeddings.
- **[PyMuPDF (fitz)](https://pymupdf.readthedocs.io/):** 📚 For reading PDF files and converting them to images.
- **[Tesseract OCR](https://github.com/tesseract-ocr/tesseract):** 🖼️ For extracting text from image-based PDFs.
- **[Google Text-to-Speech (gTTS)](https://pypi.org/project/gTTS/):** 🔊 For converting answers to audio.

### 🚀 How It Works
1. **Upload PDFs**: 🗂️ Users upload PDF documents via the interface.
2. **Text Extraction**: 📜 The content of the PDF is extracted. For **text-based PDFs**, the text is directly parsed, while **image-based PDFs** are processed with **OCR**.
3. **Embedding & Vector Store**: 🧬 The extracted text is chunked and embeddings are generated using the **SentenceTransformer** model. These embeddings are stored in **Chroma** for fast similarity search.
4. **Query Processing**: 🧐 Users can ask questions related to the PDF. The app retrieves the most relevant text chunks based on similarity search.
5. **LLM-Powered Response**: 🤖 The **Groq**-powered language model analyzes the relevant chunks and provides a comprehensive answer.
6. **Text-to-Speech**: 🔊 The response is converted into audio using **gTTS**, allowing users to listen to the answer.

### 📝 Requirements
To run this application, ensure you have the following Python packages installed:

```bash
pip install streamlit pymupdf pytesseract pillow pypdf2 langchain langchain-groq chromadb torch transformers gtts
```

💬 Feel free to send in any suggestions or feedback to improve PDFgpt. We're always looking to enhance the user experience!
