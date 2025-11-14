import os
from typing import List
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import Config


class IngestionPipeline:

    def __init__(self):
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP
        self.data_dir = Config.DATA_DIR
    
    def load_documents(self) -> List:

        """Loads text files from the data directory"""
        print(f"Loading documents from: {self.data_dir}")

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Directory {self.data_dir} not found")
        
        # Define a glob pattern to match text files
        loader = DirectoryLoader(self.data_dir, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()

        print(f"Loaded {len(documents)} documents")
        return documents
    
    def split_documents(self, docs: List) -> List:

        """Splits documents into smaller chunks"""
        print(f"Splitting documents (Size: {self.chunk_size}, Overlap: {self.chunk_overlap})")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            # First try to split by double newlines, then single newlines, then spaces, then characters
            # This is like waterfall, first it tries to keep entire paragraphs together, but if there are more than chunk_size
            # characters in a paragraph, it will split by single newlines, and so on.
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = text_splitter.split_documents(docs)
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def run(self):

        """Executes the full ingestion process"""
        raw_docs = self.load_documents()
        chunks = self.split_documents(raw_docs)
        return chunks