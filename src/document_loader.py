"""
Document Loader for RAG System
Loads documents, splits into chunks, and prepares for embedding
"""

from pathlib import Path
from typing import List, Optional
import os

# LangChain document loaders for different file types
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import config


class DocumentLoader:
    """
    Handles loading and processing documents for RAG.

    Think of this as a document prep kitchen:
    1. Load raw documents (ingredients)
    2. Clean and split them (chopping)
    3. Prepare for embedding (plating)
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        """
        Initialize document loader with chunking parameters.

        Args:
            chunk_size: Number of characters per chunk (default from config)
            chunk_overlap: Number of overlapping characters (default from config)

        Why overlap? Prevents cutting sentences/context in half.
        Example: "The refund policy is... [CHUNK 1 ENDS]
                  [CHUNK 2 STARTS] ...30 days from purchase"
        Without overlap, "30 days" loses context!
        """
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

        # RecursiveCharacterTextSplitter tries to split on:
        # 1. Paragraphs (\n\n)
        # 2. Sentences (\n)
        # 3. Words (space)
        # 4. Characters (last resort)
        # This preserves meaning better than just cutting every N chars
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,  # Count characters, not tokens
            separators=["\n\n", "\n", ". ", " ", ""],  # Split priority order
        )

        if config.DEBUG:
            print(f"📄 DocumentLoader initialized:")
            print(f"   Chunk size: {self.chunk_size}")
            print(f"   Chunk overlap: {self.chunk_overlap}")

    def load_file(self, file_path: Path) -> List[Document]:
        """
        Load a single file and return as Document objects.

        Args:
            file_path: Path to the file

        Returns:
            List of Document objects (usually 1 doc per file before splitting)

        Raises:
            ValueError: If file type is not supported
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine loader based on file extension
        suffix = file_path.suffix.lower()

        try:
            if suffix == ".pdf":
                # PyPDFLoader reads PDFs page by page
                loader = PyPDFLoader(str(file_path))
            elif suffix in [".txt", ".md"]:
                # TextLoader for plain text and markdown
                loader = TextLoader(str(file_path), encoding="utf-8")
            else:
                raise ValueError(
                    f"Unsupported file type: {suffix}. "
                    f"Supported types: .pdf, .txt, .md"
                )

            documents = loader.load()

            if config.DEBUG:
                print(f"✅ Loaded {file_path.name}: {len(documents)} page(s)")

            return documents

        except Exception as e:
            print(f"❌ Error loading {file_path.name}: {e}")
            raise

    def load_directory(self, directory_path: Path) -> List[Document]:
        """
        Load all supported documents from a directory.

        Args:
            directory_path: Path to directory containing documents

        Returns:
            List of all loaded Document objects

        Why load by directory? Easier for users to upload multiple docs at once.
        """
        directory_path = Path(directory_path)

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        documents = []
        supported_extensions = [".pdf", ".txt", ".md"]

        # Walk through directory and load each supported file
        for file_path in directory_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    docs = self.load_file(file_path)
                    documents.extend(docs)
                except Exception as e:
                    print(f"⚠️  Skipping {file_path.name}: {e}")
                    continue

        print(f"📚 Loaded {len(documents)} document(s) from {directory_path.name}")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.

        Args:
            documents: List of Document objects to split

        Returns:
            List of Document objects (chunks)

        Why split? Claude can't process 100-page PDFs at once.
        Smaller chunks = more precise retrieval.

        Example:
        Input:  1 document (10,000 chars)
        Output: 10 chunks (1,000 chars each with 100 char overlap)
        """
        if not documents:
            print("⚠️  No documents to split")
            return []

        # Calculate total characters before splitting
        total_chars = sum(len(doc.page_content) for doc in documents)

        # Split using RecursiveCharacterTextSplitter
        chunks = self.text_splitter.split_documents(documents)

        if config.DEBUG:
            print(f"✂️  Split {len(documents)} documents into {len(chunks)} chunks")
            print(f"   Total characters: {total_chars:,}")
            print(f"   Average chunk size: {total_chars // len(chunks) if chunks else 0} chars")

        return chunks

    def load_and_split(self, path: Path) -> List[Document]:
        """
        Convenience method: Load and split in one step.

        Args:
            path: Path to file or directory

        Returns:
            List of Document chunks ready for embedding

        This is the main method you'll use:
        chunks = loader.load_and_split("data/sample_docs")
        """
        path = Path(path)

        # Determine if path is file or directory
        if path.is_file():
            documents = self.load_file(path)
        elif path.is_dir():
            documents = self.load_directory(path)
        else:
            raise ValueError(f"Path is neither file nor directory: {path}")

        # Split into chunks
        chunks = self.split_documents(documents)

        print(f"✅ Processed {path.name}: {len(chunks)} chunks ready")
        return chunks

    def get_chunk_preview(self, chunks: List[Document], num_chunks: int = 3) -> None:
        """
        Print preview of first N chunks (for debugging/verification).

        Args:
            chunks: List of Document chunks
            num_chunks: Number of chunks to preview

        Why preview? Helps you verify chunks look reasonable.
        """
        print(f"\n📋 Preview of first {num_chunks} chunks:")
        print("=" * 60)

        for i, chunk in enumerate(chunks[:num_chunks]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Source: {chunk.metadata.get('source', 'Unknown')}")
            print(f"Content preview:\n{chunk.page_content[:200]}...")
            print("-" * 60)


# Example usage (for testing)
if __name__ == "__main__":
    """
    Test the document loader with sample documents.
    Run with: python -m src.document_loader
    """
    print("🧪 Testing DocumentLoader...")

    # Initialize loader
    loader = DocumentLoader()

    # Check if sample docs exist
    if not config.SAMPLE_DOCS_DIR.exists() or not any(config.SAMPLE_DOCS_DIR.iterdir()):
        print(f"\n⚠️  No documents found in {config.SAMPLE_DOCS_DIR}")
        print("Create some sample documents first!")
    else:
        # Load and split documents
        chunks = loader.load_and_split(config.SAMPLE_DOCS_DIR)

        # Show preview
        if chunks:
            loader.get_chunk_preview(chunks, num_chunks=2)
