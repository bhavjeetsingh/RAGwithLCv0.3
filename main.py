from langchain_community.document_loaders import PYPDFLoader,TextLoader, CSVLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

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