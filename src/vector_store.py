"""
Vector Store Management using FAISS
Handles embedding creation, storage, and similarity search
"""

from pathlib import Path
from typing import List, Optional
import pickle

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from src.config import config


class VectorStore:
    """
    Manages FAISS vector store for document embeddings.

    Think of this as a specialized database that stores "meaning" instead of text.

    How it works:
    1. Text chunk → OpenAI API → vector (1536 numbers representing meaning)
    2. Store vectors in FAISS (Facebook AI Similarity Search)
    3. Query → vector → FAISS finds most similar stored vectors → return original chunks

    Why vectors? "return policy" and "refund guidelines" use different words but have
    similar meaning. Vector similarity captures this semantic relationship!
    """

    def __init__(self, embedding_model: Optional[str] = None):
        """
        Initialize vector store with embedding model.

        Args:
            embedding_model: OpenAI embedding model name (default from config)

        Why OpenAI embeddings?
        - High quality (captures semantic meaning well)
        - 1536 dimensions (good balance of precision and speed)
        - Cheap ($0.02 per 1M tokens ≈ $0.002 per 1000 pages)
        """
        self.embedding_model_name = embedding_model or config.EMBEDDING_MODEL

        # Initialize OpenAI embeddings
        # This creates a client that will call OpenAI's API to convert text → vectors
        # The API key is picked up from OPENAI_API_KEY environment variable automatically
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model_name
        )

        # Vector store (will be None until created or loaded)
        self.vector_store: Optional[FAISS] = None

        if config.DEBUG:
            print(f"VectorStore initialized with {self.embedding_model_name}")

    def create_from_documents(self, documents: List[Document]) -> FAISS:
        """
        Create FAISS vector store from documents.

        Args:
            documents: List of Document chunks (from DocumentLoader)

        Returns:
            FAISS vector store instance

        What happens here:
        1. Each document chunk is sent to OpenAI API
        2. OpenAI returns a 1536-dimensional vector (list of 1536 numbers)
        3. FAISS creates an index that organizes these vectors for fast search
        4. Metadata (source file, page, etc.) is stored alongside vectors

        Example:
        "What is your return policy?"
        → [0.023, -0.145, 0.891, ..., 0.234] (1536 numbers)

        "Our return policy allows..."
        → [0.019, -0.142, 0.887, ..., 0.229] (very similar numbers!)
        """
        if not documents:
            raise ValueError("Cannot create vector store from empty document list")

        print(f"Creating embeddings for {len(documents)} chunks...")
        print(f"   This will cost ~${len(documents) * 0.00001:.4f} (OpenAI API)")

        # Create FAISS vector store from documents
        # This calls OpenAI API for each chunk and builds the FAISS index
        try:
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )

            print(f"Vector store created with {len(documents)} vectors")
            return self.vector_store

        except Exception as e:
            print(f"Error creating vector store: {e}")
            raise

    def save(self, path: Optional[Path] = None) -> None:
        """
        Save FAISS index and embeddings to disk.

        Args:
            path: Directory to save to (defaults to config.VECTOR_STORE_DIR)

        Saves two files:
        1. index.faiss - The FAISS index (vectors and search structure)
        2. index.pkl - Metadata (document text, source files, etc.)

        Why save? You don't want to pay OpenAI every time you restart the app!
        Once you create embeddings, save them and reuse them.
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save. Create one first with create_from_documents()")

        save_path = path or config.VECTOR_STORE_DIR
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            # FAISS.save_local saves both the index and metadata
            self.vector_store.save_local(str(save_path))
            print(f"Vector store saved to {save_path}")

        except Exception as e:
            print(f"Error saving vector store: {e}")
            raise

    def load(self, path: Optional[Path] = None) -> FAISS:
        """
        Load FAISS index from disk.

        Args:
            path: Directory to load from (defaults to config.VECTOR_STORE_DIR)

        Returns:
            Loaded FAISS vector store

        Why load? Much faster than recreating embeddings!
        - Creating embeddings: ~10 seconds + API cost
        - Loading from disk: ~0.1 seconds, free
        """
        load_path = path or config.VECTOR_STORE_DIR
        load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"Vector store not found at {load_path}")

        # Check if required files exist
        index_file = load_path / "index.faiss"
        pkl_file = load_path / "index.pkl"

        if not index_file.exists() or not pkl_file.exists():
            raise FileNotFoundError(
                f"Vector store files missing at {load_path}. "
                f"Expected: index.faiss and index.pkl"
            )

        try:
            # Load the vector store with the same embeddings model
            self.vector_store = FAISS.load_local(
                str(load_path),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True  # Required for pickle loading
            )

            print(f"Vector store loaded from {load_path}")
            return self.vector_store

        except Exception as e:
            print(f"Error loading vector store: {e}")
            raise

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Search for documents similar to query.

        Args:
            query: User's question or search query
            k: Number of results to return (default from config)

        Returns:
            List of most similar Document chunks

        How it works:
        1. Convert query to vector: "What's your return policy?" → [0.02, -0.14, ...]
        2. FAISS compares query vector to all stored vectors (uses cosine similarity)
        3. Returns top k most similar chunks
        4. These chunks become context for Claude!

        Cosine similarity example:
        - Query vector: [1, 0, 0]
        - Chunk A vector: [0.9, 0.1, 0] → similarity = 0.99 (very similar!)
        - Chunk B vector: [0, 1, 0] → similarity = 0.10 (not similar)
        """
        if self.vector_store is None:
            raise ValueError("No vector store loaded. Create or load one first.")

        k = k or config.RETRIEVAL_K

        try:
            # Perform similarity search
            # This converts query to vector and finds k nearest neighbors
            results = self.vector_store.similarity_search(
                query=query,
                k=k
            )

            if config.DEBUG:
                print(f"Found {len(results)} results for query: '{query[:50]}...'")

            return results

        except Exception as e:
            print(f"Error during similarity search: {e}")
            raise

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[tuple[Document, float]]:
        """
        Search with similarity scores.

        Args:
            query: User's question
            k: Number of results

        Returns:
            List of (Document, score) tuples
            Score is distance (lower = more similar)

        Use this when you want to see HOW similar results are.
        Helpful for debugging: "Why did it retrieve this chunk?"
        """
        if self.vector_store is None:
            raise ValueError("No vector store loaded. Create or load one first.")

        k = k or config.RETRIEVAL_K

        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )

            if config.DEBUG:
                print(f"Search results with scores:")
                for i, (doc, score) in enumerate(results, 1):
                    print(f"   {i}. Score: {score:.4f} | Source: {doc.metadata.get('source', 'Unknown')}")

            return results

        except Exception as e:
            print(f"Error during similarity search: {e}")
            raise

    def get_retriever(self, k: Optional[int] = None):
        """
        Get a LangChain retriever interface.

        Args:
            k: Number of documents to retrieve

        Returns:
            VectorStoreRetriever object

        Why a retriever? LangChain chains (like RAG) expect a "retriever" interface.
        This wraps our vector store in that interface.

        In Phase 4, we'll use this:
        retriever = vector_store.get_retriever()
        rag_chain = RetrievalQA.from_chain_type(llm=claude, retriever=retriever)
        """
        if self.vector_store is None:
            raise ValueError("No vector store loaded. Create or load one first.")

        k = k or config.RETRIEVAL_K

        return self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )


# Example usage and testing
if __name__ == "__main__":
    """
    Test the vector store with sample documents.
    Run with: PYTHONIOENCODING=utf-8 python -m src.vector_store
    """
    from src.document_loader import DocumentLoader

    print("Testing VectorStore...")
    print("=" * 60)

    # Step 1: Load documents
    print("\nStep 1: Loading documents...")
    loader = DocumentLoader()
    chunks = loader.load_and_split(config.SAMPLE_DOCS_DIR)

    # Step 2: Create vector store
    print(f"\nStep 2: Creating embeddings for {len(chunks)} chunks...")
    vector_store = VectorStore()
    vector_store.create_from_documents(chunks)

    # Step 3: Save to disk
    print("\nStep 3: Saving vector store...")
    vector_store.save()

    # Step 4: Test similarity search
    print("\nStep 4: Testing similarity search...")
    print("=" * 60)

    test_queries = [
        "What is your return policy?",
        "How long does shipping take?",
        "Can I get a refund?",
        "Do you accept PayPal?"
    ]

    for query in test_queries:
        print(f"\n❓ Query: '{query}'")
        results = vector_store.similarity_search(query, k=2)

        for i, doc in enumerate(results, 1):
            source = Path(doc.metadata.get('source', 'Unknown')).name
            preview = doc.page_content[:150].replace('\n', ' ')
            print(f"   {i}. [{source}] {preview}...")

    print("\n" + "=" * 60)
    print("Vector store test complete!")
    print(f"Index saved to: {config.VECTOR_STORE_DIR}")
