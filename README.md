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
     
     - Superior handling of documents with images, tables, and complex layouts
     - Enhanced visual understanding and context preservation
     - Better performance with detailed technical documents
     - *Requires higher GPU resources (Recommended: NVIDIA GPU with ≥16GB VRAM)*
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


## 📞 Contact
- Tharun Reddy Pyayala
- Email: pyayala@umich.edu
- LinkedIn: [Tharun Reddy Pyayala](https://www.linkedin.com/in/tharun-reddy-pyayala)
