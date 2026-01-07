"""
Complete RAG Chain using Pinecone Vector Database and GROQ LLM 
Production-ready implementation with error handling.
"""
# Import libraries

import os
import time
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List, Any
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader

###########################################################################################
### 1. CONFIGURATION CLASS
###########################################################################################

class RAGConfig:
    """ Configuration for RAG System """
    load_dotenv()
    # API Keys
    PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
    GROQ_API_KEY=os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

    # Pinecone Configuration
    INDEX_NAME="rag-knowledge-base"
    CLOUD="aws"
    REGION="us-east-1"

    # Embedding Configuration
    USE_OPENAI_EMBEDDINGS=False
    OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
    # HUGGINGFACE_EMBEDDING_MODEL="sentence_transformers/all-MiniLM-L6-v2"
    HUGGINGFACE_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

    # Chunking Configuration
    CHUNK_SIZE=1000
    CHUNK_OVERLAP=350

    # Retrieval Configuration
    TOP_K_RESULTS=3
    SEARCH_TYPE='similarity'

    # LLM Configuration
    GROQ_MODEL="llama-3.3-70b-versatile"
    TEMPERATURE=0
    MAX_TOKENS=1024

    # Document Processing
    DOCUMENT_PATH="./data"
    SUPPORTED_EXTENSIONS=["txt", "pdf", "md"]
###########################################################################################
### 2. RAG SYSTEM CLASS
###########################################################################################

class RAGSystem:
    """
    Complete RAG System with Pinecone and Groq LLM
    """

    def __init__(self, config: RAGConfig=None):
        """ Initialize RAG System """
        self.config=config or RAGConfig()
        self._validate_config()

        print("Initializing RAG System...")

        self.embeddings=self._initialize_embeddings()
        self.pc=self._initialize_pinecone()
        self.llm=self._initialize_llm()
        self.vectorstore=None
        self.retriever=None
        self.rag_chain=None

        print("RAG System initialized successfully!")
    
    def _validate_config(self):
        """ Validate configuration """
        if "xxxxx" in self.config.PINECONE_API_KEY:
            raise ValueError("Please set your PINECONE_API_KEY")
        if "xxxxx" in self.config.GROQ_API_KEY:
            raise ValueError("Please set your GROQ_API_KEY")
        if self.config.USE_OPENAI_EMBEDDINGS and "xxxxx" in self.config.OPENAI_API_KEY:
            raise ValueError("Pleaset set your OPENAI_API_KEY or use HuggingFace Embeddings")
    
    def _initialize_embeddings(self):
        """ Initialize embeddings """
        if self.config.USE_OPENAI_EMBEDDINGS:
            print("Using OpenAI Embeddings...")
            embeddings=OpenAIEmbeddings(
                model=self.config.OPENAI_EMBEDDING_MODEL,
                openai_api_key=self.config.OPENAI_API_KEY
            )
            self.embedding_dimension=1536
        
        else:
            print("Using HuggingFace embeddings(local, free)...")
            embeddings = HuggingFaceEmbeddings(
                model_name=self.config.HUGGINGFACE_EMBEDDING_MODEL
            )
            self.embedding_dimension=384
        
        return embeddings
    
    def _initialize_pinecone(self):
        """ Initialize Pinecone client and index """
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
                time.sleep(2)
            print(f"Index {self.config.INDEX_NAME} is ready")
        else:
            print(f"Index {self.config.INDEX_NAME} already exists!")
        
        return pc
    def _initialize_llm(self):
        """ Initialize Groq LLM """
        return ChatGroq(
            model=self.config.GROQ_MODEL,
            temperature=self.config.TEMPERATURE,
            max_tokens=self.config.MAX_TOKENS,
            groq_api_key=self.config.GROQ_API_KEY
        )
    def load_documents(self, path: str = None, file_type: str = 'txt') -> List[Document]:
        """ Load Documents from directory or file """
        path=path or self.config.DOCUMENT_PATH

        # Create directory if does not exists
        Path(path).mkdir(parents=True,exist_ok=True)

        print(f"Loading documents from : {path}")

        try:
            if file_type == "txt":
                loader=DirectoryLoader(
                    path=path,
                    glob="**/*.txt",
                    loader_cls=TextLoader,
                    show_progress=True,
                    loader_kwargs={'encoding':'utf-8'}
                )
            elif file_type == 'pdf':
                loader=DirectoryLoader(
                    path=path,
                    glob="**/*.pdf",
                    loader_cls=PyPDFLoader,
                    show_progress=True
                )
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            documents=loader.load()
            print(f"Loaded {len(documents)} documents.")
            return documents
        except Exception as e:
            print(f"Error while loading documents: {e}")
            return []
    
    def create_sample_documents(self) -> List[Document]:
        """ Create sample documents for testing """
        print(f"\nCreating Sample Documents...")
        
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
        documents = [Document(page_content=text, metadata={"source":f"sample_{i+1}"}) 
                     for i, text in enumerate(sample_texts) ]
        print(f"Created {len(documents)} sample documents...")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """ Split documents into chunks """
        print(f"Splitting document into chunks...")

        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n","\n"," ", ""]
        )
        chunks = text_splitter.split_documents(documents=documents)
        print(f"Created {len(chunks)} chunks.")
        return chunks
    
    def create_vectorstore(self, documents: List[Document]):
        """ Create Pinecone vectorstore """
        print(f"\n Creating vector store in Pinecone...")

        try:
            self.vectorstore=PineconeVectorStore.from_documents(
                documents=documents,
                embedding=self.embeddings,
                index_name=self.config.INDEX_NAME,
            )
            print(f"Added {len(documents)} to Pinecone vectorstore... ")
        except Exception as e:
            print(f"Error creating vector store: {e}")
            raise
    
    def load_existing_vectorstore(self):
        """ Load existing Pinecone index """
        print(f"\n Loading existing vectorstore index...")

        self.vectorstore=PineconeVectorStore(
            index_name=self.config.INDEX_NAME,
            embedding=self.embeddings
        )
        print(f"Loaded vector store: {self.config.INDEX_NAME}")
    
    def create_retriever(self):
        """ Create retriever from vector store """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call your create_vectorstore() first")
        if self.config.SEARCH_TYPE=="similarity":
            self.retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k":self.config.TOP_K_RESULTS}
            )
        elif self.config.SEARCH_TYPE=="mmr":
            self.retriever=self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs= {"k": self.config.TOP_K_RESULTS,
                               "fetch_k":10,
                               "lambda_mult":0.5
                               }
            )
        print(f"Created retriever {type(self.retriever)} with search type: {self.config.SEARCH_TYPE}")
    
    def build_rag_chain(self):
        """ Build the RAG chain """
        if not self.retriever:
            raise ValueError("Retriever not initialized. Call create_retriever() first.")
        print("\n Building RAG Chain...")

        # Create prompt template
        prompt_template = """You are a helpful AI assistant. Answer the question based on the provided context.
        If you cannot find the answer in the context, say "I don't have enough information to answer that question."
        Be concise and accurate in your responses. 
        
        Context: 
        {context}

        Question: {question}
        Answer:"""

        prompt=ChatPromptTemplate.from_template(prompt_template)

        # Format retrieved context
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Build RAG Chain using LCEL
        self.rag_chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        print(f"RAG Chain build successful!")
    
    def query(self, question: str) -> Dict[str, Any]:
        """ Query RAG Chain """
        if not self.rag_chain:
            raise ValueError("RAG Chain not built. Call build_rag_chain() first.")
        
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
                    "content": doc.page_content[:200]+"...",
                    "metadata": doc.metadata 
                }
                for doc in retrieved_docs
            ]
        }

        print(f"\nAnswer: {answer}")
        print(f"\nSources used: {len(retrieved_docs)} documents")

        return result
    
    def batch_queries(self, questions: List[str]) -> List[Dict[str, Any]]:
        """ Process multiple queries """
        print("\n"+'='*60)
        print(f"Processing {len(questions)} queries...")
        print("="*60)

        results=[]
        for question in questions:
            result = self.query(question=question)
            results.append(result)
        
        return results
    
    def similarity_search(self, query:str, k:int=3, with_scores: bool=False):
        """ Perform direct similarity search """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        print("\n"+'='*60)
        print(f"Similarity Search: {query}")
        print("="*60)

        if with_scores:
            results = self.vectorstore.similarity_search_with_score(query,k=k)
            for doc, score in results:
                print(f"\nScore: {score:.4f}")
                print(f"Content: {doc.page_content[:200]}...")
                print(f"Metadata: {doc.metadata}")
        else:
            results = self.vectorstore.similarity_search(query, k=k)
            for i, doc in enumerate(results):
                print(f"\nResult {i}:")
                print(f"Content: {doc.page_content[:200]}...")
                print(f"Metadata: {doc.metadata}")
        return results
    
    def get_index_stats(self) -> Dict[str, Any]:
        """ Get Pinecone index statistics """
        index=self.pc.Index(self.config.INDEX_NAME)
        stats=index.describe_index_stats()

        print("\n"+"="*60)
        print("Index Statistics:")
        print("="*60)
        print(f"\nIndex name: {self.config.INDEX_NAME}")
        print(f"Total Vectors: {stats['total_vector_count']}")
        print(f"Dimension: {stats['dimension']}")
        print(f"Index fullness: {stats.get('index_fullness','N/A')}")

        return stats
    
    def add_documents(self, documents: List[Document]):
        """ Add more documents to existing index """
        if not self.vectorstore:
            self.load_existing_vectorstore()
        
        chunks = self.split_documents(documents=documents)
        self.vectorstore.add_documents(chunks)
        print(f"Added {len(chunk)} chunks to the vectorstore")

    def clear_index(self):
        """ Clear all vectors from index """
        index=self.pc.Index(self.config.INDEX_NAME)
        index.delete(delete_all=True)
        print(f"All vectors cleared from index {self.config.INDEX_NAME}")
    
    def delete_index(self):
        """ Delete the Pinecone index """
        self.pc.delete_index(self.config.INDEX_NAME)
        print(f"Index {self.config.INDEX_NAME} is deleted")

###########################################################################################
### 3. MAIN EXECUTION
###########################################################################################

def main():
    """ Main function execution """

    print("\n"+"="*60)
    print("RAG System + Pinecone + Groq LLM")
    print("="*60)

    try:
        # Initialize RAGSystem
        rag = RAGSystem()

        print("\n Create Sample Documents:")
        documents = rag.create_sample_documents()

        # Split documents
        chunks = rag.split_documents(documents=documents)

        # Create retriever and build RAG Chain
        rag.create_vectorstore(chunks)
        rag.create_retriever()
        rag.build_rag_chain()

        # Get Index statistics
        rag.get_index_stats()

        # Test queries
        print("\n"+"="*60)
        print("Testing RAG system")
        print("="*60)

        # Single query
        result = rag.query("What is Machine Learning?")

        # Multiple queries
        questions = [
            'What is deep learning?',
            'Explain neural network',
            'What is the difference between supervised and unsupervised learning?'
        ]

        batch_results=rag.batch_queries(questions=questions)

        # Similarity search with scores
        rag.similarity_search(
            "artificial intelligence applications",
            k=3,
            with_scores=True
        )

        # RAG Index Statistics
        rag.get_index_stats()

        # Delete RAG Index
        rag.delete_index()

        print("\n"+"="*60)
        print("RAG system ready!")
        print("="*60)

        print("\nYou can now:")
        print("1. use rag.query('your question') for single queries")
        print("2. Use rag.batch_queries('your questions list') for multiple questions")
        print("3. Use rag.similarity_search('your phrase') for finding similarity score")
        print("4. Use rag.add_documents() to add more content")
        print("5. Use rag.get_index_stats() for getting statistics of the index")

    except Exception as e:
        print(f"\nConfiguration error: {e}")
        print("Please set your API keys:")
        print("1. Set PINECONE_API_KEY environment variable or update RAGConfig")
        print("2. Set GROQ_API_KEY environment variable or update RAGConfig")
        print("3. Optionally set OPENAI_API_KEY environment variable if using OPENAI Embeddings")
        import traceback
        traceback.print_exc()

###########################################################################################
### 4. INTERACTIVE MODE OF EXECUTION
###########################################################################################

def interactive_mode():
    """ Run RAG system in interactive mode """

    print("\n"+"="*60)
    print("RAG SYSTEM - INTERACTIVE MODE")
    print("="*60)
    
    try:
        rag = RAGSystem()

        # Setup
        documents=rag.create_sample_documents()
        chunks=rag.split_documents(documents=documents)
        rag.create_vectorstore(chunks)
        rag.create_retriever()
        rag.build_rag_chain()

        print("\n System is ready. Type 'quit' to exit.\n")

        while True:
            question=input("Type your question: ").strip()

            if question.lower() in ['quit','exit','q']:
                print("Goodbye!")
                break

            if not question:
                continue

            result=rag.query(question=question)
            print()

    except Exception as e:
        import traceback
        traceback.print_exc()
###########################################################################################
### 5. RUN THE SYSTEM
###########################################################################################

if __name__=="__main__":
    main()





