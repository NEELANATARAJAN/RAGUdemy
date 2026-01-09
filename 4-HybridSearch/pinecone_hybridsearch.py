"""
Complete RAG Chain using Pinecone Vector Database and Groq LLM
Production-ready implementation with error handling and examples

INSTALLATION:
pip install langchain langchain-pinecone langchain-groq langchain-openai \
    langchain-huggingface langchain-community pinecone-client pypdf rank-bm25

IMPORTANT for EnsembleRetriever:
- Install rank-bm25: pip install rank-bm25
- EnsembleRetriever combines BM25 (keyword) + Vector (semantic) search

USAGE:
python rag_system.py              # Run main demo
python rag_system.py ensemble     # Demo Ensemble Retriever
python rag_system.py interactive  # Interactive Q&A mode
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# ============================================================================
# 1. CONFIGURATION CLASS
# ============================================================================

class RAGConfig:
    """Configuration for RAG system"""
    
    # API Keys (replace with your actual keys)
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_xxxxx")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "gsk_xxxxx")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-xxxxx")
    
    # Pinecone Configuration
    INDEX_NAME = "rag-knowledge-base"
    CLOUD = "aws"
    REGION = "us-east-1"
    
    # Embedding Configuration
    USE_OPENAI_EMBEDDINGS = False  # Set to True to use OpenAI, False for HuggingFace
    OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
    HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Chunking Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval Configuration
    TOP_K_RESULTS = 3
    SEARCH_TYPE = "similarity"  # Options: "similarity", "mmr", "ensemble"
    USE_ENSEMBLE = False  # Set to True to use Ensemble Retriever (BM25 + Vector)
    ENSEMBLE_WEIGHTS = [0.5, 0.5]  # [BM25 weight, Vector weight]
    
    # LLM Configuration
    GROQ_MODEL = "llama-3.3-70b-versatile"
    TEMPERATURE = 0
    MAX_TOKENS = 1024
    
    # Document Processing
    DOCUMENTS_PATH = "./documents"
    SUPPORTED_EXTENSIONS = ["txt", "pdf", "md"]

# ============================================================================
# 2. RAG SYSTEM CLASS
# ============================================================================

class RAGSystem:
    """Complete RAG System with Pinecone and Groq"""
    
    def __init__(self, config: RAGConfig = None):
        """Initialize RAG System"""
        self.config = config or RAGConfig()
        self._validate_config()
        
        print("Initializing RAG System...")
        
        # Initialize components
        self.embeddings = self._initialize_embeddings()
        self.pc = self._initialize_pinecone()
        self.llm = self._initialize_llm()
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.documents_for_bm25 = []  # Store documents for BM25 retriever
        
        print("✓ RAG System initialized successfully")
    
    def _validate_config(self):
        """Validate configuration"""
        if "xxxxx" in self.config.PINECONE_API_KEY:
            raise ValueError("Please set your PINECONE_API_KEY")
        if "xxxxx" in self.config.GROQ_API_KEY:
            raise ValueError("Please set your GROQ_API_KEY")
        if self.config.USE_OPENAI_EMBEDDINGS and "xxxxx" in self.config.OPENAI_API_KEY:
            raise ValueError("Please set your OPENAI_API_KEY or use HuggingFace embeddings")
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        if self.config.USE_OPENAI_EMBEDDINGS:
            print("Using OpenAI embeddings...")
            embeddings = OpenAIEmbeddings(
                model=self.config.OPENAI_EMBEDDING_MODEL,
                openai_api_key=self.config.OPENAI_API_KEY
            )
            self.embedding_dimension = 1536
        else:
            print("Using HuggingFace embeddings (local, free)...")
            embeddings = HuggingFaceEmbeddings(
                model_name=self.config.HUGGINGFACE_EMBEDDING_MODEL
            )
            self.embedding_dimension = 384
        
        return embeddings
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client and index"""
        pc = Pinecone(api_key=self.config.PINECONE_API_KEY)
        
        # Create index if it doesn't exist
        if self.config.INDEX_NAME not in pc.list_indexes().names():
            print(f"Creating new index: {self.config.INDEX_NAME}")
            pc.create_index(
                name=self.config.INDEX_NAME,
                dimension=self.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self.config.CLOUD,
                    region=self.config.REGION
                )
            )
            
            # Wait for index to be ready
            print("Waiting for index to be ready...")
            while not pc.describe_index(self.config.INDEX_NAME).status['ready']:
                time.sleep(1)
            print(f"✓ Index {self.config.INDEX_NAME} is ready!")
        else:
            print(f"✓ Using existing index: {self.config.INDEX_NAME}")
        
        return pc
    
    def _initialize_llm(self):
        """Initialize Groq LLM"""
        return ChatGroq(
            model=self.config.GROQ_MODEL,
            temperature=self.config.TEMPERATURE,
            max_tokens=self.config.MAX_TOKENS,
            groq_api_key=self.config.GROQ_API_KEY
        )
    
    def load_documents(self, path: str = None, file_type: str = "txt") -> List[Document]:
        """Load documents from directory or file"""
        path = path or self.config.DOCUMENTS_PATH
        
        # Create directory if it doesn't exist
        Path(path).mkdir(parents=True, exist_ok=True)
        
        print(f"\nLoading documents from: {path}")
        
        try:
            if file_type == "txt":
                loader = DirectoryLoader(
                    path=path,
                    glob="**/*.txt",
                    loader_cls=TextLoader,
                    show_progress=True,
                    loader_kwargs={'encoding': 'utf-8'}
                )
            elif file_type == "pdf":
                loader = DirectoryLoader(
                    path=path,
                    glob="**/*.pdf",
                    loader_cls=PyPDFLoader,
                    show_progress=True
                )
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            documents = loader.load()
            print(f"✓ Loaded {len(documents)} documents")
            return documents
        
        except Exception as e:
            print(f"Error loading documents: {e}")
            return []
    
    def create_sample_documents(self) -> List[Document]:
        """Create sample documents for testing"""
        print("\nCreating sample documents...")
        
        sample_texts = [
            """Machine Learning is a subset of artificial intelligence that enables systems to 
            learn and improve from experience without being explicitly programmed. It focuses on 
            developing computer programs that can access data and use it to learn for themselves.""",
            
            """Deep Learning is a subset of machine learning that uses neural networks with 
            multiple layers. These neural networks attempt to simulate the behavior of the human 
            brain, allowing it to learn from large amounts of data.""",
            
            """Natural Language Processing (NLP) is a branch of artificial intelligence that helps 
            computers understand, interpret and manipulate human language. NLP draws from many 
            disciplines including computer science and computational linguistics.""",
            
            """Computer Vision is a field of artificial intelligence that trains computers to 
            interpret and understand the visual world. Using digital images from cameras and videos 
            and deep learning models, machines can accurately identify and classify objects.""",
            
            """Reinforcement Learning is a type of machine learning where an agent learns to make 
            decisions by performing actions in an environment to maximize cumulative reward. The 
            agent learns through trial and error.""",
            
            """Neural Networks are computing systems inspired by biological neural networks that 
            constitute animal brains. A neural network consists of interconnected groups of nodes 
            called artificial neurons.""",
            
            """Supervised Learning is a machine learning approach where the model is trained on 
            labeled data. The algorithm learns from the training dataset by iteratively making 
            predictions and adjusting based on correct answers.""",
            
            """Unsupervised Learning is a type of machine learning that looks for previously 
            undetected patterns in a dataset with no pre-existing labels. The algorithm must 
            figure out what is being shown."""
        ]
        
        documents = [Document(page_content=text, metadata={"source": f"sample_{i}"}) 
                    for i, text in enumerate(sample_texts)]
        
        print(f"✓ Created {len(documents)} sample documents")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        print(f"\nSplitting documents into chunks...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"✓ Created {len(chunks)} chunks")
        return chunks
    
    def create_vectorstore(self, documents: List[Document]):
        """Create and populate vector store"""
        print(f"\nCreating vector store in Pinecone...")
        
        # Store documents for potential BM25 retriever
        self.documents_for_bm25 = documents
        
        try:
            self.vectorstore = PineconeVectorStore.from_documents(
                documents=documents,
                embedding=self.embeddings,
                index_name=self.config.INDEX_NAME
            )
            print(f"✓ Added {len(documents)} documents to Pinecone")
        except Exception as e:
            print(f"Error creating vector store: {e}")
            raise
    
    def load_existing_vectorstore(self):
        """Load existing Pinecone index"""
        print(f"\nLoading existing vector store...")
        
        self.vectorstore = PineconeVectorStore(
            index_name=self.config.INDEX_NAME,
            embedding=self.embeddings
        )
        print(f"✓ Loaded vector store: {self.config.INDEX_NAME}")
    
    def create_retriever(self):
        """Create retriever from vector store"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call create_vectorstore() first.")
        
        print(f"\nCreating retriever...")
        
        if self.config.USE_ENSEMBLE:
            # Create Ensemble Retriever (BM25 + Vector Search)
            self.retriever = self._create_ensemble_retriever()
        elif self.config.SEARCH_TYPE == "similarity":
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.config.TOP_K_RESULTS}
            )
            print(f"✓ Created similarity retriever")
        elif self.config.SEARCH_TYPE == "mmr":
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": self.config.TOP_K_RESULTS,
                    "fetch_k": 10,
                    "lambda_mult": 0.5
                }
            )
            print(f"✓ Created MMR retriever")
    
    def _create_ensemble_retriever(self):
        """Create Ensemble Retriever combining BM25 and Vector Search"""
        print("Creating Ensemble Retriever (BM25 + Vector Search)...")
        
        try:
            # Create BM25 retriever (keyword-based)
            bm25_retriever = BM25Retriever.from_documents(self.documents_for_bm25)
            bm25_retriever.k = self.config.TOP_K_RESULTS
            
            # Create vector retriever (semantic)
            vector_retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.TOP_K_RESULTS}
            )
            
            # Combine with ensemble
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=self.config.ENSEMBLE_WEIGHTS
            )
            
            print(f"✓ Created Ensemble Retriever (BM25: {self.config.ENSEMBLE_WEIGHTS[0]}, Vector: {self.config.ENSEMBLE_WEIGHTS[1]})")
            return ensemble_retriever
            
        except Exception as e:
            print(f"⚠ Error creating Ensemble Retriever: {e}")
            print("Falling back to vector-only retriever")
            return self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.TOP_K_RESULTS}
            )
    
    def create_contextual_compression_retriever(self):
        """Create retriever with contextual compression"""
        if not self.retriever:
            self.create_retriever()
        
        print("\nCreating Contextual Compression Retriever...")
        
        try:
            compressor = LLMChainExtractor.from_llm(self.llm)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.retriever
            )
            
            self.retriever = compression_retriever
            print("✓ Created Contextual Compression Retriever")
            
        except Exception as e:
            print(f"⚠ Error creating compression retriever: {e}")
            print("Using base retriever without compression")
    
    def build_rag_chain(self):
        """Build the RAG chain"""
        if not self.retriever:
            raise ValueError("Retriever not initialized. Call create_retriever() first.")
        
        print("\nBuilding RAG chain...")
        
        # Create prompt template
        prompt_template = """You are a helpful AI assistant. Answer the question based on the provided context.
If you cannot find the answer in the context, say "I don't have enough information to answer that question."
Be concise and accurate in your responses.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Define format function
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Build chain using LCEL
        self.rag_chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("✓ RAG chain built successfully")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system"""
        if not self.rag_chain:
            raise ValueError("RAG chain not built. Call build_rag_chain() first.")
        
        print(f"\n{'='*60}")
        print(f"Query: {question}")
        print('='*60)
        
        # Get retrieved documents
        retrieved_docs = self.retriever.invoke(question)
        
        # Get answer
        answer = self.rag_chain.invoke(question)
        
        result = {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                }
                for doc in retrieved_docs
            ]
        }
        
        print(f"\nAnswer: {answer}")
        print(f"\nSources used: {len(retrieved_docs)} documents")
        
        return result
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries"""
        print(f"\n{'='*60}")
        print(f"Processing {len(questions)} queries...")
        print('='*60)
        
        results = []
        for question in questions:
            result = self.query(question)
            results.append(result)
        
        return results
    
    def similarity_search(self, query: str, k: int = 3, with_scores: bool = False):
        """Perform direct similarity search"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized.")
        
        print(f"\n{'='*60}")
        print(f"Similarity Search: {query}")
        print('='*60)
        
        if with_scores:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            for doc, score in results:
                print(f"\nScore: {score:.4f}")
                print(f"Content: {doc.page_content[:200]}...")
                print(f"Metadata: {doc.metadata}")
        else:
            results = self.vectorstore.similarity_search(query, k=k)
            for i, doc in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"Content: {doc.page_content[:200]}...")
                print(f"Metadata: {doc.metadata}")
        
        return results
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics"""
        index = self.pc.Index(self.config.INDEX_NAME)
        stats = index.describe_index_stats()
        
        print(f"\n{'='*60}")
        print("Index Statistics")
        print('='*60)
        print(f"Index Name: {self.config.INDEX_NAME}")
        print(f"Total Vectors: {stats['total_vector_count']}")
        print(f"Dimension: {stats['dimension']}")
        print(f"Index Fullness: {stats.get('index_fullness', 'N/A')}")
        
        return stats
    
    def add_documents(self, documents: List[Document]):
        """Add more documents to existing index"""
        if not self.vectorstore:
            self.load_existing_vectorstore()
        
        chunks = self.split_documents(documents)
        self.vectorstore.add_documents(chunks)
        print(f"✓ Added {len(chunks)} new chunks to index")
    
    def clear_index(self):
        """Clear all vectors from index"""
        index = self.pc.Index(self.config.INDEX_NAME)
        index.delete(delete_all=True)
        print(f"✓ All vectors cleared from index {self.config.INDEX_NAME}")
    
    def delete_index(self):
        """Delete the Pinecone index"""
        self.pc.delete_index(self.config.INDEX_NAME)
        print(f"✓ Index {self.config.INDEX_NAME} deleted")

# ============================================================================
# 3. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*60)
    print("RAG SYSTEM - Pinecone + Groq LLM")
    print("="*60)
    
    try:
        # Initialize RAG system
        rag = RAGSystem()
        
        # Option 1: Use sample documents (for testing)
        print("\n[Option 1: Using sample documents]")
        documents = rag.create_sample_documents()
        
        # Option 2: Load from directory (uncomment to use)
        # print("\n[Option 2: Loading from directory]")
        # documents = rag.load_documents(path="./documents", file_type="txt")
        
        # If no documents, use samples
        if not documents:
            print("No documents found, using sample documents...")
            documents = rag.create_sample_documents()
        
        # Split documents
        chunks = rag.split_documents(documents)
        
        # Create vector store
        rag.create_vectorstore(chunks)
        
        # Create retriever and build chain
        rag.create_retriever()
        rag.build_rag_chain()
        
        # Get index statistics
        rag.get_index_stats()
        
        # Test queries
        print("\n" + "="*60)
        print("TESTING RAG SYSTEM")
        print("="*60)
        
        # Single query
        result = rag.query("What is machine learning?")
        
        # Multiple queries
        questions = [
            "What is deep learning?",
            "Explain neural networks",
            "What is the difference between supervised and unsupervised learning?"
        ]
        
        batch_results = rag.batch_query(questions)
        
        # Similarity search with scores
        rag.similarity_search(
            "artificial intelligence applications",
            k=3,
            with_scores=True
        )
        
        print("\n" + "="*60)
        print("RAG SYSTEM READY!")
        print("="*60)
        print("\nRetriever Configuration:")
        print(f"  Type: {'Ensemble (BM25 + Vector)' if rag.config.USE_ENSEMBLE else rag.config.SEARCH_TYPE}")
        if rag.config.USE_ENSEMBLE:
            print(f"  Weights: BM25={rag.config.ENSEMBLE_WEIGHTS[0]}, Vector={rag.config.ENSEMBLE_WEIGHTS[1]}")
        print(f"  Top K Results: {rag.config.TOP_K_RESULTS}")
        print("\nYou can now:")
        print("1. Use rag.query('your question') for single queries")
        print("2. Use rag.batch_query(['q1', 'q2']) for batch processing")
        print("3. Use rag.similarity_search('query') for direct search")
        print("4. Use rag.add_documents() to add more content")
        print("5. Use rag.get_index_stats() to check index status")
        
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nPlease set your API keys:")
        print("1. Set PINECONE_API_KEY environment variable or update RAGConfig")
        print("2. Set GROQ_API_KEY environment variable or update RAGConfig")
        print("3. Optionally set OPENAI_API_KEY if using OpenAI embeddings")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# 4. ENSEMBLE RETRIEVER DEMO
# ============================================================================

def demo_ensemble_retriever():
    """Demonstrate Ensemble Retriever capabilities"""
    
    print("\n" + "="*60)
    print("ENSEMBLE RETRIEVER DEMONSTRATION")
    print("="*60)
    print("\nComparing: BM25 (keyword) vs Vector (semantic) vs Ensemble (both)")
    
    try:
        # Initialize with Ensemble enabled
        config = RAGConfig()
        config.USE_ENSEMBLE = True
        config.ENSEMBLE_WEIGHTS = [0.5, 0.5]  # Equal weight
        
        rag = RAGSystem(config)
        
        # Create sample documents
        documents = rag.create_sample_documents()
        chunks = rag.split_documents(documents)
        rag.create_vectorstore(chunks)
        
        # Test query
        test_query = "neural networks for deep learning"
        
        print(f"\n{'='*60}")
        print(f"Test Query: '{test_query}'")
        print('='*60)
        
        # 1. BM25 Only (keyword search)
        print("\n[1] BM25 Retriever (Keyword-based):")
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 3
        bm25_results = bm25_retriever.invoke(test_query)
        for i, doc in enumerate(bm25_results, 1):
            print(f"  {i}. {doc.page_content[:100]}...")
        
        # 2. Vector Only (semantic search)
        print("\n[2] Vector Retriever (Semantic-based):")
        vector_retriever = rag.vectorstore.as_retriever(search_kwargs={"k": 3})
        vector_results = vector_retriever.invoke(test_query)
        for i, doc in enumerate(vector_results, 1):
            print(f"  {i}. {doc.page_content[:100]}...")
        
        # 3. Ensemble (combined)
        print("\n[3] Ensemble Retriever (BM25 + Vector):")
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )
        ensemble_results = ensemble_retriever.invoke(test_query)
        for i, doc in enumerate(ensemble_results, 1):
            print(f"  {i}. {doc.page_content[:100]}...")
        
        print("\n" + "="*60)
        print("Ensemble Retriever combines keyword matching (BM25)")
        print("with semantic understanding (Vector Search) for better results!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error in demo: {e}")
        print("\nMake sure you have installed: pip install rank-bm25")
        import traceback
        traceback.print_exc()

# ============================================================================
# 5. INTERACTIVE MODE
# ============================================================================

def interactive_mode():
    """Run RAG system in interactive mode"""
    
    print("\n" + "="*60)
    print("RAG SYSTEM - INTERACTIVE MODE")
    print("="*60)
    
    try:
        rag = RAGSystem()
        
        # Setup
        documents = rag.create_sample_documents()
        chunks = rag.split_documents(documents)
        rag.create_vectorstore(chunks)
        rag.create_retriever()
        rag.build_rag_chain()
        
        print("\n✓ System ready! Type 'quit' to exit.\n")
        
        while True:
            question = input("Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            result = rag.query(question)
            print()
    
    except Exception as e:
        print(f"Error: {e}")

# ============================================================================
# 6. RUN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "ensemble":
            # Demo ensemble retriever
            demo_ensemble_retriever()
        elif mode == "interactive":
            # Run interactive mode
            interactive_mode()
        else:
            print("Usage: python rag_system.py [ensemble|interactive]")
            print("  ensemble    - Demo Ensemble Retriever")
            print("  interactive - Interactive Q&A mode")
            print("  (no args)   - Run main demo")
    else:
        # Run main demo
        main()
    
    # Uncomment to always run specific mode:
    # demo_ensemble_retriever()
    # interactive_mode()