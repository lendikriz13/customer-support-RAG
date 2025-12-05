"""
AI Customer Support Agent - Streamlit Web Interface
Beautiful UI with custom design system
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import time
from typing import List

from src.config import config
from src.document_loader import DocumentLoader
from src.vector_store import VectorStore
from src.rag_chain import RAGChain


# Page configuration
st.set_page_config(
    page_title="TechStyle AI Support",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with simplified, reliable design
def load_css():
    """Inject custom CSS - simplified version that works reliably"""
    st.markdown("""
    <style>
    /* Import Google Fonts - Space Grotesk & Inter */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    /* Root variables - Design System Colors */
    :root {
        --color-beige: #e7e3dc;
        --color-navy: #1E3A8A;
        --color-cyan: #06B6D4;
        --color-navy-light: #3B82F6;
        --font-heading: 'Space Grotesk', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        --font-body: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* Main app background - simple gradient */
    .stApp {
        background: linear-gradient(135deg, #e7e3dc 0%, #d4cfc5 50%, #e7e3dc 100%);
        font-family: var(--font-body);
    }

    /* Hide sidebar completely */
    [data-testid="stSidebar"] {
        display: none;
    }

    /* Remove sidebar collapse button */
    button[kind="header"] {
        display: none;
    }

    /* Ensure content is readable */
    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Typography - Headings */
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: var(--font-heading) !important;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--color-navy);
        filter: drop-shadow(0 10px 8px rgba(0, 0, 0, 0.04)) drop-shadow(0 4px 3px rgba(0, 0, 0, 0.1));
    }

    h1 {
        font-size: 4.5rem;
        line-height: 1.1;
    }

    h2 {
        font-size: 2rem;
        line-height: 1.2;
    }

    h3 {
        font-size: 1.125rem;
        line-height: 1.3;
    }

    /* Body text */
    p, .stMarkdown p, .stText {
        font-family: var(--font-body);
        font-size: 1rem;
        line-height: 1.5;
        color: var(--color-navy);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(231, 227, 220, 0.95);
        backdrop-filter: blur(12px);
        border-right: 2px solid rgba(30, 58, 138, 0.1);
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        font-family: var(--font-heading);
        color: var(--color-navy);
    }

    /* Badge component */
    .badge {
        display: inline-block;
        background-color: rgba(231, 227, 220, 0.1);
        border: 2px solid rgba(231, 227, 220, 0.3);
        backdrop-filter: blur(12px);
        border-radius: 9999px;
        padding: 0.5rem 1.25rem;
        font-family: var(--font-heading);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.75rem;
        color: var(--color-cyan);
        margin-bottom: 1rem;
    }

    /* Chat messages */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border-radius: 1rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(30, 58, 138, 0.1);
    }

    [data-testid="stChatMessageContent"] {
        font-family: var(--font-body);
        font-size: 1rem;
        line-height: 1.6;
        color: var(--color-navy);
    }

    /* User message */
    [data-testid="stChatMessage"][data-testid*="user"] {
        background-color: rgba(6, 182, 212, 0.1) !important;
        border: 1px solid rgba(6, 182, 212, 0.3);
    }

    /* Assistant message */
    [data-testid="stChatMessage"][data-testid*="assistant"] {
        background-color: rgba(255, 255, 255, 0.95) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--color-navy) 0%, #1e40af 100%);
        color: var(--color-beige);
        font-family: var(--font-heading);
        font-weight: 600;
        border: none;
        border-radius: 0.75rem;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        letter-spacing: 0.025em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(30, 58, 138, 0.2);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(30, 58, 138, 0.3);
        background: linear-gradient(135deg, #1e40af 0%, var(--color-navy) 100%);
    }

    /* Text input */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: rgba(255, 255, 255, 0.9);
        border: 2px solid rgba(30, 58, 138, 0.2);
        border-radius: 0.75rem;
        font-family: var(--font-body);
        color: var(--color-navy);
        padding: 0.75rem;
        font-size: 1rem;
    }

    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
        border-color: var(--color-cyan);
        box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.1);
    }

    /* Chat input - lighter supporting color */
    /* Target the bottom fixed container */
    section[data-testid="stBottom"] {
        background-color: transparent !important;
        background: transparent !important;
    }

    section[data-testid="stBottom"] > div {
        background-color: transparent !important;
        background: transparent !important;
    }

    .stBottomBlockContainer {
        background-color: transparent !important;
        background: transparent !important;
    }

    [data-testid="stBottom"] {
        background-color: transparent !important;
        background: transparent !important;
    }

    /* All chat input containers */
    [data-testid="stChatInput"] {
        background-color: transparent !important;
    }

    [data-testid="stChatInput"] > div {
        background-color: transparent !important;
        border-radius: 1rem;
        padding: 0;
    }

    [data-testid="stChatInput"] > div > div {
        background-color: transparent !important;
    }

    .stChatInputContainer {
        background-color: transparent !important;
    }

    /* Override any default dark backgrounds */
    div[class*="stChatInput"] {
        background-color: transparent !important;
    }

    [data-testid="stChatInput"] textarea {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border: 2px solid rgba(6, 182, 212, 0.3) !important;
        border-radius: 0.75rem !important;
        font-family: var(--font-body) !important;
        color: var(--color-navy) !important;
        padding: 0.75rem !important;
        font-size: 1rem !important;
    }

    [data-testid="stChatInput"] textarea:focus {
        border-color: var(--color-cyan) !important;
        box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.15) !important;
        outline: none !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.8);
        border: 2px dashed rgba(30, 58, 138, 0.3);
        border-radius: 1rem;
        padding: 2rem;
    }

    /* Info/Success/Warning boxes */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 0.75rem;
        border-left: 4px solid var(--color-cyan);
        font-family: var(--font-body);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 0.75rem;
        font-family: var(--font-heading);
        color: var(--color-navy);
        font-weight: 600;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: var(--font-heading);
        font-size: 2rem;
        font-weight: 700;
        color: var(--color-navy);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Hide anchor link buttons on headers */
    .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a {
        display: none !important;
    }

    h1 a, h2 a, h3 a {
        display: none !important;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }

    ::-webkit-scrollbar-track {
        background: var(--color-beige);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--color-navy);
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--color-cyan);
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'rag_chain' not in st.session_state:
        try:
            st.session_state.rag_chain = RAGChain()
            st.session_state.vector_store_loaded = True
            st.session_state.load_error = None
        except Exception as e:
            st.session_state.rag_chain = None
            st.session_state.vector_store_loaded = False
            st.session_state.load_error = str(e)

    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False


def sidebar():
    """Render customer-facing sidebar"""
    with st.sidebar:
        st.markdown('<div class="badge">AI SUPPORT AGENT</div>', unsafe_allow_html=True)
        st.title("💬 Support Chat")

        st.markdown("---")

        # Help section
        st.subheader("How can we help?")
        st.markdown("""
        I can answer questions about:
        - Return & refund policies
        - Shipping options & delivery times
        - Payment methods
        - Order cancellation
        - And more!
        """)

        st.markdown("---")

        # Status indicator
        if st.session_state.vector_store_loaded:
            st.success("🟢 AI Assistant Online")
        else:
            st.error("🔴 AI Assistant Offline")

        st.markdown("---")

        # Clear chat
        if st.button("🗑️ New Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")

        # Footer
        st.caption("Powered by Claude AI")
        st.caption("© 2024 TechStyle Electronics")


def process_documents(uploaded_files: List):
    """Process uploaded documents and create/update vector store"""
    with st.spinner("Processing documents..."):
        try:
            # Save uploaded files
            config.SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)

            for uploaded_file in uploaded_files:
                file_path = config.SAMPLE_DOCS_DIR / uploaded_file.name
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

            # Load and process documents
            loader = DocumentLoader()
            chunks = loader.load_and_split(config.SAMPLE_DOCS_DIR)

            st.info(f"📄 Loaded {len(chunks)} chunks from {len(uploaded_files)} file(s)")

            # Create vector store
            vector_store = VectorStore()
            vector_store.create_from_documents(chunks)
            vector_store.save()

            # Reload RAG chain
            st.session_state.rag_chain = RAGChain()
            st.session_state.vector_store_loaded = True
            st.session_state.documents_processed = True

            st.success(f"✅ Successfully processed {len(uploaded_files)} document(s)!")
            time.sleep(1)
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error processing documents: {e}")


def main():
    """Main app"""
    load_css()
    initialize_session_state()

    # Header
    st.markdown('<div class="badge">#52WeeksOfAI - Agent #2</div>', unsafe_allow_html=True)
    st.title("🤖 TechStyle AI Support")
    st.markdown("**Intelligent customer support powered by RAG (Retrieval Augmented Generation)**")

    # Check if vector store is loaded
    if not st.session_state.vector_store_loaded:
        st.warning("⚠️ **No knowledge base found.** Please upload documents in the sidebar to get started.")

        # Show error if there was one during loading
        if hasattr(st.session_state, 'load_error') and st.session_state.load_error:
            with st.expander("🔍 Debug Info - Click to see error details"):
                st.error(f"Error loading RAG chain: {st.session_state.load_error}")

        # Quick start example
        with st.expander("📖 Quick Start Guide"):
            st.markdown("""
            **Welcome to TechStyle AI Support!**

            This AI assistant answers customer questions using your company documentation.

            **To get started:**
            1. Upload your documents (PDFs, TXT, MD files) in the sidebar
            2. Click "Process Documents" to build the knowledge base
            3. Start asking questions in the chat!

            **Example questions you can ask:**
            - What is your return policy?
            - How long does shipping take?
            - Do you accept PayPal?
            - Can I cancel my order?
            """)
        return

    # Chat interface
    st.markdown("---")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.caption(f"• {source}")

    # Chat input
    if prompt := st.chat_input("Ask a question about our policies, shipping, or products..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get answer from RAG chain with conversation history
                    result = st.session_state.rag_chain.ask_with_details(
                        prompt,
                        chat_history=st.session_state.messages
                    )

                    answer = result['answer']
                    sources = result['source_files']

                    # Display answer
                    st.markdown(answer)

                    # Display sources
                    if sources:
                        with st.expander("📚 Sources"):
                            for source in sources:
                                source_name = Path(source).name
                                st.caption(f"• {source_name}")

                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": [Path(s).name for s in sources]
                    })

                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


if __name__ == "__main__":
    main()
