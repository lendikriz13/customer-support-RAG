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
    initial_sidebar_state="expanded"
)

# Custom CSS with your design system
def load_css():
    """Inject custom CSS based on the provided design system"""
    st.markdown("""
    <style>
    /* Import Google Fonts - Space Grotesk & Inter */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    /* Root variables - Design System Colors */
    :root {
        --color-beige: #e7e3dc;
        --color-navy: #1E3A8A;
        --color-cyan: #06B6D4;
        --font-heading: 'Space Grotesk', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        --font-body: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* Main app background with gradient blobs */
    .stApp {
        background-color: var(--color-beige);
        font-family: var(--font-body);
        position: relative;
    }

    /* Gradient blob background */
    .stApp::before {
        content: '';
        position: fixed;
        top: 20%;
        left: 15%;
        width: 600px;
        height: 600px;
        background: radial-gradient(circle, var(--color-navy) 0%, transparent 70%);
        border-radius: 50%;
        filter: blur(120px);
        opacity: 0.9;
        transform: translate(-50%, -50%) rotate(12deg);
        z-index: 0;
        pointer-events: none;
    }

    .stApp::after {
        content: '';
        position: fixed;
        top: -10%;
        right: -10%;
        width: 1100px;
        height: 900px;
        background: radial-gradient(circle, var(--color-navy) 0%, transparent 70%);
        border-radius: 50%;
        filter: blur(100px);
        opacity: 1.0;
        z-index: 0;
        pointer-events: none;
    }

    /* Noise texture overlay */
    .stApp > div:first-child {
        position: relative;
    }

    .stApp > div:first-child::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E");
        mix-blend-mode: overlay;
        opacity: 0.3;
        z-index: 1;
        pointer-events: none;
    }

    .stApp > div:first-child::after {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 300 300' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E");
        mix-blend-mode: overlay;
        opacity: 0.5;
        z-index: 1;
        pointer-events: none;
    }

    /* Ensure content is above background */
    .main .block-container {
        position: relative;
        z-index: 2;
        max-width: 816px;
        padding-top: 3rem;
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
        except:
            st.session_state.rag_chain = None
            st.session_state.vector_store_loaded = False

    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False


def sidebar():
    """Render sidebar with document management"""
    with st.sidebar:
        st.markdown('<div class="badge">AI SUPPORT AGENT</div>', unsafe_allow_html=True)
        st.title("📚 Knowledge Base")

        st.markdown("---")

        # Document upload
        st.subheader("Upload Documents")
        st.caption("Add PDF, TXT, or MD files to the knowledge base")

        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'txt', 'md'],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        if uploaded_files:
            if st.button("🔄 Process Documents", use_container_width=True):
                process_documents(uploaded_files)

        st.markdown("---")

        # Current status
        st.subheader("📊 Status")

        if st.session_state.vector_store_loaded:
            st.success("✅ Vector store loaded")
        else:
            st.warning("⚠️ No vector store found")

        # Show document stats
        if config.SAMPLE_DOCS_DIR.exists():
            docs = list(config.SAMPLE_DOCS_DIR.glob("*"))
            doc_files = [d for d in docs if d.is_file() and d.suffix in ['.pdf', '.txt', '.md']]
            st.metric("Documents", len(doc_files))

        st.markdown("---")

        # Settings
        with st.expander("⚙️ Settings"):
            st.slider("Temperature", 0.0, 1.0, config.TEMPERATURE, 0.1,
                     help="Higher = more creative, Lower = more focused")
            st.slider("Retrieved Chunks", 1, 10, config.RETRIEVAL_K, 1,
                     help="Number of document chunks to retrieve")

        st.markdown("---")

        # Clear chat
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


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
    sidebar()

    # Header
    st.markdown('<div class="badge">#52WeeksOfAI - Agent #2</div>', unsafe_allow_html=True)
    st.title("🤖 TechStyle AI Support")
    st.markdown("**Intelligent customer support powered by RAG (Retrieval Augmented Generation)**")

    # Check if vector store is loaded
    if not st.session_state.vector_store_loaded:
        st.warning("⚠️ **No knowledge base found.** Please upload documents in the sidebar to get started.")

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
                    # Get answer from RAG chain
                    result = st.session_state.rag_chain.ask_with_details(prompt)

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
