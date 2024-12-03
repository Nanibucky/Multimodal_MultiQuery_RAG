from langchain.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import HumanMessage, SystemMessage
from langchain import hub
from langchain.load import dumps, loads
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from bs4 import BeautifulSoup
import os
from langchain.schema import Document
import PyPDF2
import tempfile
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import logging
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Tuple
import backoff
from dotenv import load_dotenv
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Constants
MAX_FILE_SIZE_MB = 10
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

class InputTracker:
    """Manages input change detection and vector store update requirements"""
    
    @staticmethod
    def get_input_hash(input_source) -> Optional[str]:
        """Generate a hash for the current input"""
        try:
            if isinstance(input_source, str):  # URL input
                return hashlib.md5(input_source.encode()).hexdigest()
            elif hasattr(input_source, 'read'):  # File input
                content = input_source.read()
                input_source.seek(0)  # Reset file pointer
                return hashlib.md5(content).hexdigest()
            return None
        except Exception as e:
            logger.error(f"Error generating input hash: {e}")
            return None

    @staticmethod
    def has_input_changed(new_hash: Optional[str], last_hash: Optional[str]) -> bool:
        """Check if input has changed from last processed input"""
        if not new_hash:
            return False
        return new_hash != last_hash

class AudioTranscriber:
    """Handles audio transcription using Whisper model"""
    
    def __init__(self):
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def transcribe_audio(self, audio_file):
        try:
            audio_input, sample_rate = torchaudio.load(audio_file)
            
            # Handle stereo to mono conversion
            if audio_input.shape[0] > 1:
                audio_input = audio_input.mean(dim=0, keepdim=True)
            
            # Resample to 16kHz if necessary
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                audio_input = resampler(audio_input)
            
            input_features = self.processor(
                audio_input.squeeze(0), 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features
            
            predicted_ids = self.model.generate(input_features)
            return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        except Exception as e:
            logger.error(f"Error in audio transcription: {str(e)}")
            raise

class DataProcessor:
    """Handles processing of different input types and document preparation"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        self.audio_transcriber = AudioTranscriber()
        self.input_tracker = InputTracker()

    def validate_file_size(self, file_size: int) -> bool:
        """Check if file size is within acceptable limits"""
        return file_size <= (MAX_FILE_SIZE_MB * 1024 * 1024)

    def process_input(self, input_source, input_type: str) -> Tuple[List[Document], str]:
        """Process input and return documents along with input hash"""
        input_hash = InputTracker.get_input_hash(input_source)
        docs = self.process_data(
            input_source,
            is_pdf=input_type == "pdf",
            is_audio=input_type == "audio"
        )
        return docs, input_hash

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def process_data(self, input_source, is_pdf=False, is_audio=False) -> List[Document]:
        """Process different types of input data"""
        try:
            if is_pdf:
                return self._process_pdf(input_source)
            elif is_audio:
                return self._process_audio(input_source)
            else:
                return self._process_url(input_source)
        except Exception as e:
            logger.error(f"Error in data processing: {str(e)}")
            raise

    def _process_pdf(self, pdf_file) -> List[Document]:
        """Process PDF input"""
        if not self.validate_file_size(pdf_file.size):
            raise ValueError(f"PDF file size exceeds {MAX_FILE_SIZE_MB}MB limit")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(pdf_file.read())

        try:
            reader = PyPDF2.PdfReader(temp_file_path)
            text = ' '.join(page.extract_text() for page in reader.pages)
            documents = [Document(page_content=text)]
            return self.text_splitter.split_documents(documents)
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def _process_audio(self, audio_file) -> List[Document]:
        """Process audio input"""
        if not self.validate_file_size(audio_file.size):
            raise ValueError(f"Audio file size exceeds {MAX_FILE_SIZE_MB}MB limit")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(audio_file.read())

        try:
            transcription = self.audio_transcriber.transcribe_audio(temp_file_path)
            documents = [Document(page_content=transcription)]
            return self.text_splitter.split_documents(documents)
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def _process_url(self, url: str) -> List[Document]:
        """Process URL input"""
        loader = WebBaseLoader(url)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)

class VectorStoreManager:
    """Manages FAISS vector store operations"""
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.store_path = "faiss-index"

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def create_or_update(self, docs: List[Document]) -> FAISS:
        """Create or update the vector store with new documents"""
        try:
            vectorstore = FAISS.from_documents(docs, self.embeddings)
            self._save_store(vectorstore)
            return vectorstore
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise

    def _save_store(self, vectorstore: FAISS):
        """Save vector store to disk"""
        vectorstore.save_local(self.store_path)

    def load(self) -> Optional[FAISS]:
        """Load vector store from disk"""
        try:
            return FAISS.load_local(self.store_path, self.embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return None

class ResponseGenerator:
    """Handles response generation using the RAG pipeline"""
    
    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def generate_response(self, question: str) -> str:
        """Generate response using RAG pipeline"""
        try:
            # Generate alternative queries
            alternative_queries = self._generate_queries(question)
            
            # Retrieve and process documents
            all_docs = self._retrieve_documents(alternative_queries)
            unique_docs = self._remove_duplicates(all_docs)
            
            # Generate response
            return self._get_llm_response(question, unique_docs)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def _generate_queries(self, question: str) -> List[str]:
        """Generate alternative versions of the query"""
        chain = QUERY_PROMPT | self.llm | parser
        result = chain.invoke({"question": question})
        return [q.strip() for q in result.split("\n") if q.strip()]

    def _retrieve_documents(self, queries: List[str]) -> List[Document]:
        """Retrieve relevant documents for all query versions"""
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        all_docs = []
        for query in queries:
            docs = retriever.get_relevant_documents(query)
            all_docs.extend(docs)
        return all_docs

    def _remove_duplicates(self, docs: List[Document]) -> List[Document]:
        """Remove duplicate documents"""
        hashable_docs = [dumps(doc) for doc in docs]
        return [loads(doc) for doc in set(hashable_docs)]

    def _get_llm_response(self, question: str, docs: List[Document]) -> str:
        """Generate final response using LLM"""
        context = "\n\n".join([doc.page_content for doc in docs])
        messages = [
            SystemMessage(
                content="""You are a specialized AI assistant focusing on accurate information retrieval and presentation. 
        Your primary directive is to provide answers solely based on the given context."""
            ),
            HumanMessage(
                content=f"""[CONTEXT INFORMATION]
        ----------------
        {context}
        ----------------

        [QUERY]
        {question}

        [RESPONSE REQUIREMENTS]
        1. Answer Format:
        - Begin with a clear, direct answer to the question
        - Support with evidence from the context
        - Use bullet points for listing information
        - Include relevant quotes when applicable

        2. Context Usage:
        - Only use information present in the provided context
        - Do not incorporate external knowledge
        - If information is missing, state: "The provided context does not contain information about [specific aspect]"

        3. Information Organization:
        - Present information in order of relevance
        - Group related information together
        - Use clear transitions between different points

        4. Technical Information (if applicable):
        - Format code snippets in code blocks
        - Explain technical terms found in the context
        - Maintain technical accuracy

        5. Verification Steps:
        - Ensure each point is directly from the context
        - Verify relevance to the question
        - Confirm no external information is added

        Please provide your response following these guidelines while maintaining professionalism and clarity.""")
        ]
        return self.llm(messages).content

# Initialize LLM and parser
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.5,
    openai_api_key=OPENAI_API_KEY
)
emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
parser = StrOutputParser()

# Improved query prompt template
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Generate five different versions 
    of the given question to help retrieve relevant documents. Focus on:
    1. Different phrasings
    2. Related concepts
    3. Specific aspects
    4. Broader context
    5. Key terms

    Original question: {question}
    Provide alternatives separated by newlines."""
)

def main():
    """Main Streamlit application"""
    st.set_page_config(page_title='Enhanced RAG System')
    st.header('Intelligent Document Analysis System 🤖')

    # Initialize managers and processors
    data_processor = DataProcessor()
    vector_store_manager = VectorStoreManager(emb)

    # Initialize session state
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "last_processed_hash" not in st.session_state:
        st.session_state.last_processed_hash = None
    if "needs_update" not in st.session_state:
        st.session_state.needs_update = False

    # Input selection
    input_choice = st.radio(
        "Choose input method:",
        ["Enter Blog URL", "Upload PDF", "Upload Audio"]
    )

    # Input handling
    input_source = None
    input_type = None
    
    if input_choice == "Enter Blog URL":
        input_source = st.text_input('Enter the URL:')
        input_type = "url"
    elif input_choice == "Upload PDF":
        input_source = st.file_uploader("Upload PDF", type="pdf")
        input_type = "pdf"
    else:
        input_source = st.file_uploader("Upload Audio", type=["wav", "mp3"])
        input_type = "audio"

    # Check for input changes
    if input_source:
        current_hash = InputTracker.get_input_hash(input_source)
        if InputTracker.has_input_changed(current_hash, st.session_state.last_processed_hash):
            st.session_state.needs_update = True
            st.warning("⚠️ Input has changed - Vector store needs to be updated before querying")

    # Vector store update section
    if st.session_state.needs_update and input_source:
        st.sidebar.markdown("### Vector Store Status")
        if st.sidebar.button('Update Vector Store'):
            try:
                with st.spinner('Processing input and updating vector store...'):
                    docs, current_hash = data_processor.process_input(input_source, input_type)
                    st.session_state.vector_store = vector_store_manager.create_or_update(docs)
                    st.session_state.last_processed_hash = current_hash
                    st.session_state.needs_update = False
                st.success('✅ Vector Store Updated Successfully!')
            except Exception as e:
                st.error(f"Error updating vector store: {str(e)}")
                return

    # Question handling
    user_question = st.text_input('Ask a question:')

# Generate Response button at the bottom
    if st.button('Generate Response', key='generate_response'):
        if not user_question:
            st.error("Please enter a question first.")
        elif st.session_state.needs_update:
            st.error("⚠️ Please update the vector store before asking questions about the new content")
        elif st.session_state.vector_store is None:
            st.error("⚠️ Please provide input and update the vector store first")
        else:
            try:
                with st.spinner('Generating response...'):
                    response_generator = ResponseGenerator(llm, st.session_state.vector_store)
                    response = response_generator.generate_response(user_question)
                    st.write(response)
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

if __name__ == '__main__':
    main()
