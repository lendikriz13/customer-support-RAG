"""
RAG Chain - Retrieval Augmented Generation
Connects vector store retrieval with Claude for context-grounded responses
"""

from typing import Dict, List, Optional
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from src.config import config
from src.vector_store import VectorStore


class RAGChain:
    """
    RAG (Retrieval Augmented Generation) Chain.

    The complete RAG flow:
    1. User asks a question
    2. Retrieve relevant chunks from vector store
    3. Send chunks + question to Claude
    4. Claude generates answer using ONLY the provided context
    5. Return grounded, accurate answer (no hallucinations!)

    Why RAG?
    - Without RAG: Claude uses training data (outdated, generic, may hallucinate)
    - With RAG: Claude uses YOUR docs (current, specific, grounded in facts)
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        retrieval_k: Optional[int] = None
    ):
        """
        Initialize RAG chain.

        Args:
            vector_store: VectorStore instance (will load from disk if None)
            model_name: Claude model name (default from config)
            temperature: LLM temperature 0-1 (default from config)
            retrieval_k: Number of chunks to retrieve (default from config)

        Temperature explained:
        - 0.0: Deterministic, consistent (best for factual Q&A)
        - 0.3: Slightly varied but focused (good for customer support)
        - 1.0: Creative, varied (good for writing, NOT for facts)
        """
        self.model_name = model_name or config.LLM_MODEL
        self.temperature = temperature if temperature is not None else config.TEMPERATURE
        self.retrieval_k = retrieval_k or config.RETRIEVAL_K

        # Initialize vector store
        if vector_store is None:
            self.vector_store = VectorStore()
            try:
                self.vector_store.load()
                if config.DEBUG:
                    print(f"Loaded vector store from {config.VECTOR_STORE_DIR}")
            except FileNotFoundError:
                print(" No vector store found. Create one first with VectorStore.create_from_documents()")
                raise
        else:
            self.vector_store = vector_store

        # Initialize Claude
        self.llm = ChatAnthropic(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=1024  # Max response length
        )

        # Create the RAG chain
        self.chain = self._create_chain()

        if config.DEBUG:
            print(f"RAG Chain initialized:")
            print(f"   Model: {self.model_name}")
            print(f"   Temperature: {self.temperature}")
            print(f"   Retrieval K: {self.retrieval_k}")

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """
        Create the prompt template for RAG.

        This is THE MOST IMPORTANT part of RAG!
        A good prompt:
        1. Tells Claude to use ONLY the provided context
        2. Instructs to say "I don't know" if answer isn't in context
        3. Keeps tone helpful and professional
        4. Asks for concise but complete answers

        Why this matters:
        Bad prompt → Claude hallucinates, makes up info
        Good prompt → Claude stays grounded, admits when unsure
        """

        template = """You are a helpful customer support assistant for TechStyle Electronics.

Your job is to answer customer questions using ONLY the information provided in the context below.

IMPORTANT RULES:
1. ONLY use information from the context provided below
2. If the answer is not in the context, say "I don't have that information in our documentation. Let me connect you with a human agent who can help."
3. Be concise but complete - answer the question fully but don't add extra information
4. Be professional and friendly
5. If you reference specific policies or details, you can mention where they came from (e.g., "According to our return policy...")
6. Use the conversation history to understand follow-up questions and pronouns (like "it", "that", "the same")

{chat_history}

Context from company documentation:
{context}

Customer Question: {question}

Your Answer:"""

        return ChatPromptTemplate.from_template(template)

    def _format_docs(self, docs: List[Document]) -> str:
        """
        Format retrieved documents into a string for the prompt.

        Args:
            docs: List of retrieved Document chunks

        Returns:
            Formatted string with all chunks

        Why format? Claude needs clean, readable context.
        We separate chunks with "---" so Claude knows where one chunk ends and another begins.
        """
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content
            formatted.append(f"[Source {i}: {source}]\n{content}")

        return "\n\n---\n\n".join(formatted)

    def _format_chat_history(self, messages: List[Dict[str, str]]) -> str:
        """
        Format conversation history for the prompt.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Formatted conversation history string
        """
        if not messages:
            return ""

        # Take last 6 messages (3 exchanges) for context
        recent_messages = messages[-6:]

        formatted = ["Previous conversation:"]
        for msg in recent_messages:
            role = "Customer" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")

        return "\n".join(formatted) + "\n"

    def _create_chain(self):
        """
        Create the RAG chain using LangChain LCEL (LangChain Expression Language).

        Chain flow:
        1. Question comes in
        2. Retriever gets relevant chunks
        3. Format chunks + question into prompt
        4. Send to Claude
        5. Parse Claude's response
        6. Return answer

        LCEL syntax explained:
        - RunnablePassthrough(): Passes the question through unchanged
        - retriever | format_docs: Get docs, then format them
        - prompt | llm | parser: Fill prompt, send to Claude, parse output
        - {}: Dictionary of values that get filled into the prompt
        """

        # Get retriever from vector store
        retriever = self.vector_store.get_retriever(k=self.retrieval_k)

        # Create prompt template
        prompt = self._create_prompt_template()

        # Output parser (converts Claude's response to string)
        output_parser = StrOutputParser()

        # Build the chain using LCEL
        # This is equivalent to: question → retrieve → format → prompt → llm → parse
        # Note: We'll pass chat_history separately in the ask() method
        chain = (
            {
                "context": lambda x: self._format_docs(retriever.invoke(x["question"])),
                "question": lambda x: x["question"],
                "chat_history": lambda x: x.get("chat_history", "")
            }
            | prompt  # Fill the prompt template
            | self.llm  # Send to Claude
            | output_parser  # Parse response to string
        )

        return chain

    def ask(self, question: str, return_sources: bool = False, chat_history: List[Dict[str, str]] = None) -> Dict[str, any]:
        """
        Ask a question and get an answer from the RAG system.

        Args:
            question: User's question
            return_sources: If True, also return the source chunks used
            chat_history: Optional list of previous messages for context

        Returns:
            Dict with 'answer' and optionally 'sources'

        This is the main method you'll use!
        Example:
            rag = RAGChain()
            result = rag.ask("What's your return policy?")
            print(result['answer'])
        """

        if config.DEBUG:
            print(f"\nQuestion: {question}")

        # Format chat history if provided
        formatted_history = self._format_chat_history(chat_history or [])

        # Get the answer from the chain
        answer = self.chain.invoke({
            "question": question,
            "chat_history": formatted_history
        })

        result = {"answer": answer}

        # Optionally get source chunks
        if return_sources:
            retriever = self.vector_store.get_retriever(k=self.retrieval_k)
            sources = retriever.invoke(question)
            result["sources"] = sources

            if config.DEBUG:
                print(f"\nRetrieved {len(sources)} source chunks:")
                for i, doc in enumerate(sources, 1):
                    source = doc.metadata.get('source', 'Unknown')
                    preview = doc.page_content[:100].replace('\n', ' ')
                    print(f"   {i}. [{source}] {preview}...")

        if config.DEBUG:
            print(f"\nAnswer: {answer}\n")

        return result

    def ask_with_details(self, question: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, any]:
        """
        Ask a question and get detailed response with sources and metadata.

        Args:
            question: User's question
            chat_history: Optional list of previous messages for context

        Returns:
            Dict with:
                - answer: Claude's response
                - sources: List of source Document chunks
                - source_files: List of unique source files used
                - chunk_count: Number of chunks retrieved

        Useful for debugging or showing users where info came from!
        """
        result = self.ask(question, return_sources=True, chat_history=chat_history)

        # Extract source file names
        source_files = list(set(
            doc.metadata.get('source', 'Unknown')
            for doc in result['sources']
        ))

        result['source_files'] = source_files
        result['chunk_count'] = len(result['sources'])

        return result


# Example usage and testing
if __name__ == "__main__":
    """
    Test the RAG chain with various questions.
    Run with: PYTHONIOENCODING=utf-8 python -m src.rag_chain
    """

    print("Testing RAG Chain with Claude Sonnet 4...")
    print("=" * 70)

    # Initialize RAG chain
    try:
        rag = RAGChain()
        print("RAG Chain initialized successfully!\n")
    except FileNotFoundError:
        print("Vector store not found. Run vector_store.py first!")
        exit(1)

    # Test questions
    test_questions = [
        # Questions that SHOULD be answered (in the docs)
        {
            "question": "What is your return policy?",
            "expect": "Should find return policy details"
        },
        {
            "question": "How long does standard shipping take?",
            "expect": "Should find 5-7 business days"
        },
        {
            "question": "Do you accept PayPal?",
            "expect": "Should find payment methods in FAQ"
        },
        {
            "question": "Can I cancel my order after placing it?",
            "expect": "Should find 1-hour cancellation window"
        },
        # Question that should NOT be answered (not in docs)
        {
            "question": "What's your phone number?",
            "expect": "Should say 'I don't know' or connect to human"
        },
        {
            "question": "Do you sell laptops?",
            "expect": "Should say info not available (not in docs)"
        }
    ]

    print("\nTesting RAG Chain with sample questions...")
    print("=" * 70)

    for i, test in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}/{len(test_questions)}")
        print(f"{'='*70}")
        print(f"❓ Question: {test['question']}")
        print(f"📝 Expected: {test['expect']}")
        print("-" * 70)

        # Get detailed answer
        result = rag.ask_with_details(test['question'])

        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources used: {result['chunk_count']} chunks from:")
        for source in result['source_files']:
            print(f"   - {source}")

    print("\n" + "=" * 70)
    print("RAG Chain testing complete!")
    print("=" * 70)

    # Interactive mode
    print("\n🎯 Try your own questions! (Type 'quit' to exit)")
    print("-" * 70)

    while True:
        try:
            question = input("\n❓ Your question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if not question:
                continue

            result = rag.ask_with_details(question)
            print(f"\nAnswer:\n{result['answer']}")
            print(f"\nUsed {result['chunk_count']} chunks from: {', '.join(result['source_files'])}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
