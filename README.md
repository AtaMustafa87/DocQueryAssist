# DocQueryAssist
This repository contains a Python-based **intelligent chatbot** designed to assist researchers by answering questions based on the content of uploaded documents. The chatbot leverages open-source tools and models to extract, process, and retrieve information from documents, making it efficient and cost-effective to run on CPU-based systems.

---

## **Features**
- 📄 **Document Processing**: Extracts and preprocesses text from PDF documents.
- 🧠 **Question Answering**: Answers user queries based on the document content using state-of-the-art NLP models.
- ⚡ **Efficient Retrieval**: Utilizes FAISS for fast similarity search over document embeddings.
- 🌐 **User-Friendly Interface**: Built with Streamlit for an intuitive and responsive user experience.
- 🛠️ **Open-Source & CPU-Compatible**: Uses open-source tools and works seamlessly without requiring GPU resources.

---

## **How It Works**
1. **Upload a Document**: The user uploads a PDF document via the chatbot interface.
2. **Preprocess the Document**: The text is extracted, split into smaller chunks, and embedded using a sentence transformer model.
3. **Search for Relevant Information**: The chatbot retrieves the most relevant chunk for the user's question using FAISS.
4. **Answer the Question**: A pre-trained question-answering model provides an accurate response based on the retrieved chunk.

---

## **Tech Stack**
- **Python**: Core programming language.
- **PyTorch**: For running transformer models.
- **Hugging Face Transformers**: For state-of-the-art Q&A models.
- **Sentence Transformers**: For generating embeddings.
- **FAISS**: For fast similarity searches.
- **Streamlit**: For creating an interactive web app.
- **pdfplumber**: For text extraction from PDF documents.

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/AtaMustafa87/DocQueryAssist.git
cd DocQueryAssist
```
### **2. Install Dependencies**
Ensure you have Python 3.10+ installed. Then, run:
```bash
pip install -r requirements.txt
```
### **3. Run the Application**
Start the chatbot interface:
```bash
streamlit run chatbot.py
```

### **4. Interact with the Chatbot**
1. Upload a PDF document via the web interface.
2. Ask questions related to the document content.
3. Receive accurate, context-aware answers.

### **5. File Structure**

.
- chatbot.py              # Main application code
- requirements.txt        # Required Python libraries
- sample_documents/       # Example PDF documents for testing

### **6. Future Enhancements**
- 🔍 Support for Multiple File Formats: Add support for Word documents and plain text files.
- 🌐 Summarization: Incorporate summarization for quicker insights into large documents.
- 🎯 Domain-Specific Fine-Tuning: Improve accuracy for specific research areas (e.g., medical or legal).
- 🚀 Cloud Deployment: Host the chatbot on a platform like Hugging Face Spaces or Render.

### **7. Contributing**
Contributions are welcome! If you'd like to enhance the chatbot, please:

1. Fork the repository.
2. Create a new branch.
3. Submit a pull request with a clear description of your changes.

### **8. License**
This project is licensed under the [MIT License](https://mit-license.org/).

**Contact**
If you have any questions or suggestions, feel free to reach out:

Email: [ata.mustafa.1987@gmail.com](mailto:ata.mustafa.1987@gmail.com)  

GitHub: [AtaMustafa87](https://github.com/AtaMustafa87)

_Start your journey with an intelligent research assistant! 🚀_
