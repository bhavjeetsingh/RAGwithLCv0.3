from langchain_community.document_loaders import PYPDFLoader,TextLoader, CSVLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
import asyncio
from langchain.callbacks import LangChainTracer
from langsmith import Client
import os
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()

class DocumentProcessor:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.documents = []
    
    def load_documents(self):
        # Load PDFs
        pdf_loader = DirectoryLoader(
            self.directory_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        
        # Load text files
        text_loader = DirectoryLoader(
            self.directory_path,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        
        # Combine all documents
        self.documents = pdf_loader.load() + text_loader.load()
        print(f"Loaded {len(self.documents)} documents")
        return self.documents
    
    def split_documents(self, chunk_size=1000, chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=True
        )
        
        splits = text_splitter.split_documents(self.documents)
        print(f"Created {len(splits)} document chunks")
        return splits
    
class VectorStoreManager:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"  # New OpenAI embedding model
        )
        self.vectorstore = None
    
    def create_vectorstore(self, documents):
        # Create vector store with metadata filtering support
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        self.vectorstore.persist()
        print(f"Vector store created with {len(documents)} documents")
        return self.vectorstore
    
class RAGChain:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.2
        )
        self.retriever = self._setup_retriever()
        self.chain = self._setup_chain()
    
    def _setup_retriever(self):
        # Base retriever with similarity search
        base_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Add contextual compression for better results
        compressor = LLMChainExtractor.from_llm(self.llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        return compression_retriever
    
    def _setup_chain(self):
        # System prompt for RAG
        system_prompt = """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, say that you don't know. 
        Keep the answer concise and relevant to the question.
        
        Context: {context}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        # Create the chain using LCEL
        question_answer_chain = create_stuff_documents_chain(
            self.llm, 
            prompt
        )
        
        rag_chain = create_retrieval_chain(
            self.retriever, 
            question_answer_chain
        )
        
        return rag_chain
    
    def setup_multi_query_retriever(self):
    # Generate multiple queries for better coverage
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=self.vectorstore.as_retriever(),
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["question"],
                template="""You are an AI assistant tasked with generating multiple search queries.
                Generate 3 different versions of the user question to retrieve relevant documents.
                Provide these alternative questions separated by newlines.
                Original question: {question}"""
            )
        )
        return multi_query_retriever
    
    def create_hybrid_retriever(self, documents):
    # BM25 for keyword search
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 3
        
        # Semantic search from vector store
        semantic_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        
        # Ensemble retriever combines both
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, semantic_retriever],
            weights=[0.5, 0.5]
        )
        
        return ensemble_retriever
    
class ConversationalRAG:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview")
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=1000,
            return_messages=True,
            memory_key="chat_history",
            output_key="answer"
        )
        
    def create_conversational_chain(self):
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(),
            memory=self.memory,
            return_source_documents=True,
            verbose=True
        )
class OptimizedRAG:
    def __init__(self):
        # Enable caching for repeated queries
        set_llm_cache(InMemoryCache())
        
    async def async_retrieve_and_generate(self, query):
        # Parallel retrieval from multiple sources
        tasks = [
            self.retrieve_from_vectorstore(query),
            self.retrieve_from_cache(query),
            self.retrieve_from_api(query)
        ]
        
        results = await asyncio.gather(*tasks)
        return self.combine_results(results)
    
    def setup_monitoring():
        # Initialize LangSmith for production monitoring
        client = Client()
        tracer = LangChainTracer(
            project_name="production-rag",
            client=client
        )
        
        return tracer
    
class ProductionRAG:
    def __init__(self, data_directory: str):
        self.data_directory = data_directory
        self.processor = DocumentProcessor(data_directory)
        self.vector_manager = VectorStoreManager()
        self.rag_chain = None
        
    def initialize(self):
        # Load and process documents
        documents = self.processor.load_documents()
        chunks = self.processor.split_documents()
        
        # Create vector store
        vectorstore = self.vector_manager.create_vectorstore(chunks)
        
        # Initialize RAG chain
        self.rag_chain = RAGChain(vectorstore)
        
        print("RAG system initialized successfully!")
        
    def query(self, question: str) -> Dict[str, Any]:
        if not self.rag_chain:
            raise ValueError("RAG system not initialized")
        
        response = self.rag_chain.chain.invoke({
            "input": question
        })
        
        return {
            "answer": response["answer"],
            "sources": [doc.metadata for doc in response["context"]]
        }
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        return [self.query(q) for q in questions]

# Usage example
if __name__ == "__main__":
    rag = ProductionRAG("./documents")
    rag.initialize()
    
    # Test query
    result = rag.query("What are the main features of LangChain v0.3?")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
