# Multi-Query RAG System with Advanced Document Processing

## Overview
An intelligent content discovery platform leveraging advanced Retrieval-Augmented Generation (RAG) for seamless information retrieval across diverse content formats. The system implements dual document processing pipelines and sophisticated audio processing capabilities, enabling comprehensive information extraction from various sources.

## 🌟 Key Features
- **Multi-Modal Content Processing**
  - Blog posts via URL extraction
  - PDF documents with text and image processing
  - Audio content transcription and analysis
- **Dual Processing Pipeline**
  - Primary: PyPDF with GPT-4 for robust text extraction
  - Advanced: ColPaLi + QWEN2-VL for enhanced handling of PDFs with images
- **Advanced RAG Implementation**
  - Multi-query approach for comprehensive retrieval
  - Maximal Marginal Relevance (MMR) for result diversity
  - LLM re-ranking for enhanced relevance
  - Intelligent serialization for duplicate elimination

## 🛠️ Technical Architecture

### Core Components
1. **Document Processing**
   - RecursiveCharacterTextSplitter for optimal text chunking
   - BeautifulSoup for web content extraction
   - PyPDF2 for PDF processing
   - Whisper & Torchaudio for audio transcription

2. **Vector Store Management**
   - FAISS for efficient similarity search
   - OpenAI embeddings for document vectorization
   - Local storage with serialization support

3. **Query Processing**
   - Multi-query generation for comprehensive coverage
   - Context-aware prompt engineering
   - Advanced error handling and retry mechanisms

4. **User Interface**
   - Streamlit-based interactive interface
   - Real-time processing feedback
   - Flexible input handling

## 🚀 Getting Started

### Prerequisites
```bash
# Install required packages
pip install -r requirements.txt

# Additional dependencies for audio processing
apt-get install ffmpeg sox
```

### Environment Setup
```bash
# Create .env file with:
OPENAI_API_KEY=your_api_key_here
```

### Running the Application
```bash
# Start the Streamlit app
streamlit run main_rag.py
```

## 💡 Usage

### Basic Usage
1. Select input method (URL/PDF/Audio)
2. Provide input content
3. Update vector store
4. Ask questions about the content

### Advanced Features
- **Multi-Query Processing**
  ```python
  # Example of generating multiple queries
  response_generator = ResponseGenerator(llm, vector_store)
  response = response_generator.generate_response(question)
  ```

- **PDF Image Processing**
  ```python
  # Using ColPaLi + QWEN2-VL
  from colpali_vlm import process_document
  results = process_document(pdf_path)
  ```

## 🎯 Performance Metrics
- Improved information retrieval accuracy across formats
- Reduced query processing time through optimized vector search
- Enhanced user experience with intuitive interface
- Successful handling of complex documents with tables and images

## 🔧 Technical Stack
- **Frontend**: Streamlit
- **Backend**: Python, LangChain
- **LLMs**: OpenAI GPT-4, QWEN2-VL
- **Vector Storage**: FAISS
- **Document Processing**: 
  - PyPDF, ColPaLi for PDF handling
  - Whisper, Torchaudio for audio
- **Additional Tools**: 
  - BeautifulSoup for web scraping
  - Backoff for retry handling
  - Logging for system monitoring

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## 🙏 Acknowledgments
- OpenAI for GPT models
- Hugging Face for transformer models
- FAISS team for vector similarity search
- ColPaLi team for PDF processing capabilities

## 📞 Contact
- Tharun Reddy Pyayala
- Email: pyayala@umich.edu
- LinkedIn: [Tharun Reddy Pyayala](https://www.linkedin.com/in/tharun-reddy-pyayala)
