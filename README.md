# AI Customer Support Agent with RAG 🤖

> Smart AI assistant that answers customer questions using your company's documentation - no hallucinations, just accurate answers grounded in your knowledge base.

## 🎯 The Problem

Small businesses need 24/7 customer support but can't afford:
- Full support team salaries ($40k-60k/person/year)
- Overnight shifts and weekend coverage
- Training new staff on company policies

Traditional chatbots give robotic, unhelpful answers. Generic AI hallucinates information.

## ✨ The Solution

An AI agent that:
- ✅ Answers questions using YOUR company documentation (RAG technology)
- ✅ Says "I don't know" when answer isn't in the knowledge base (no hallucinations)
- ✅ Maintains conversation context (remembers what you asked 3 messages ago)
- ✅ Escalates to human support when needed
- ✅ Updates knowledge base in seconds (just upload new docs)

## 🏗️ Architecture

```
User Question
    ↓
[Streamlit Chat UI]
    ↓
[Conversational Agent]
    ↓
[Vector Store Search] → Retrieves top 3 relevant chunks
    ↓
[Claude Sonnet 4] → Generates answer using only retrieved context
    ↓
Response to User
```

**Tech Stack:**
- **LangChain**: Orchestrates the RAG pipeline
- **FAISS**: Vector database (local, fast, free)
- **OpenAI Embeddings**: Converts text to vectors (~$0.002/1000 pages)
- **Claude Sonnet 4**: Generates accurate, context-grounded responses
- **Streamlit**: Beautiful web interface (no HTML/CSS needed)
- **Python 3.12**: Modern, type-safe code

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- Anthropic API key ([get one here](https://console.anthropic.com/settings/keys))

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-support-agent-rag.git
cd ai-support-agent-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### Configuration

Edit `.env` file:
```bash
OPENAI_API_KEY=sk-proj-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Running the App

```bash
# Start the Streamlit interface
streamlit run src/app.py
```

Open your browser to `http://localhost:8501`

## 📖 Usage

### 1. Upload Documents
- Click "Browse files" in the sidebar
- Upload PDFs, TXT, or MD files (company policies, FAQs, product docs)
- Click "Process Documents"

### 2. Chat with Your Knowledge Base
- Type questions in the chat input
- Agent retrieves relevant context and answers
- Conversation history is maintained automatically

### 3. Update Knowledge
- Upload new documents anytime
- Click "Process Documents" to refresh the vector store
- New information is immediately available

## 📂 Project Structure

```
ai-support-agent-rag/
├── src/
│   ├── config.py           # Environment variables & settings
│   ├── document_loader.py  # Load & chunk documents
│   ├── vector_store.py     # FAISS vector database
│   ├── rag_chain.py        # RAG retrieval & generation
│   ├── agent.py            # Conversational agent
│   └── app.py              # Streamlit web interface
├── data/
│   ├── sample_docs/        # Upload your documents here
│   └── vector_store/       # FAISS index (auto-generated)
├── docs/
│   ├── ARCHITECTURE.md     # System design deep-dive
│   └── SETUP.md            # Detailed setup instructions
├── tests/
│   └── test_agent.py       # Unit tests
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
└── README.md               # You are here
```

## 🎓 How It Works: RAG Explained

**RAG = Retrieval Augmented Generation**

Think of it like an open-book exam for AI:

1. **Indexing Phase** (happens once when you upload docs)
   - Documents are split into chunks (~1000 characters each)
   - Each chunk is converted to a vector (list of numbers) using OpenAI embeddings
   - Vectors are stored in FAISS for fast similarity search

2. **Query Phase** (happens every time user asks a question)
   - User's question is converted to a vector
   - FAISS finds the 3 most similar document chunks
   - Claude receives: question + relevant chunks as context
   - Claude generates answer using ONLY the provided context

**Why this prevents hallucinations:**
- Claude is instructed to answer ONLY from context
- If answer isn't in retrieved chunks, Claude says "I don't know"
- You can audit which chunks were used (transparency)

## 🔧 Configuration Options

Edit `.env` to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | 1000 | Characters per chunk (tune for your docs) |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks (prevents cut-off) |
| `RETRIEVAL_K` | 3 | Number of chunks to retrieve |
| `TEMPERATURE` | 0.3 | LLM creativity (0=factual, 1=creative) |
| `EMBEDDING_MODEL` | text-embedding-3-small | OpenAI embedding model |
| `LLM_MODEL` | claude-sonnet-4-20250514 | Claude model |

## 💰 Cost Estimation

**One-time costs (per document processing):**
- 100 pages of docs: ~$0.003 (less than a penny)
- 10,000 pages: ~$3

**Per-query costs:**
- OpenAI embedding (user question): ~$0.000001
- Claude Sonnet 4 response: ~$0.003-0.01 per answer

**Example:** 1,000 customer questions/month ≈ $3-10/month

Compare to human support: $3,000-5,000/month for one agent.

## 🛠️ Development Status

**✅ Phase 1: Project Setup** (COMPLETE)
- [x] Folder structure
- [x] Dependencies configured
- [x] Git initialized

**⏳ Phase 2-10: Coming Soon**
- [ ] Document processing pipeline
- [ ] FAISS vector store integration
- [ ] RAG chain implementation
- [ ] Conversational agent
- [ ] Streamlit UI
- [ ] Testing suite
- [ ] Deployment to Railway

## 🤝 Use Cases

Perfect for:
- **E-commerce**: Product info, shipping, returns
- **SaaS**: Feature docs, API reference, troubleshooting
- **Professional Services**: Client FAQs, process docs
- **Education**: Course materials, syllabus Q&A

Not ideal for:
- Real-time data (stock prices, weather)
- Complex workflows requiring multiple steps
- Sentiment analysis or creative writing

## 📚 Documentation

- [Architecture Deep-Dive](docs/ARCHITECTURE.md) (Coming soon)
- [Setup Guide](docs/SETUP.md) (Coming soon)
- [API Reference](docs/API.md) (Coming soon)

## 🐛 Troubleshooting

**"ModuleNotFoundError"**
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt`

**"OpenAI API Error"**
- Check `.env` file has correct API key
- Verify key has billing enabled at platform.openai.com

**"No results found"**
- Documents may not be processed yet
- Click "Process Documents" in sidebar
- Check `data/vector_store/` folder exists

## 📄 License

MIT License - Free to use for commercial projects

## 🙋 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ai-support-agent-rag/issues)
- **Fiverr**: [Hire me for custom AI agents](https://fiverr.com/yourusername)
- **Email**: your.email@example.com

---

**Built with ❤️ for small businesses who deserve great customer support**

*Part of my AI Agent Portfolio - Agent #2: Customer Support RAG*
