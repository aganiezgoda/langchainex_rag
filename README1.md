# RAG Tutorial with LangChain & Azure OpenAI

A hands-on tutorial demonstrating **Retrieval Augmented Generation (RAG)** using LangChain and Azure OpenAI. Perfect for students learning how to build AI applications that can answer questions based on custom documents.

## What is RAG?

RAG (Retrieval Augmented Generation) is a technique that enhances LLM responses by:

1. **Retrieving** relevant documents from a knowledge base
2. **Augmenting** the prompt with this context
3. **Generating** an answer grounded in the retrieved information

This solves key LLM limitations: hallucinations, outdated knowledge, and lack of domain-specific information.

## Architecture

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│   Document  │────▶│  Text Splitter  │────▶│  Embeddings │
│   (Web URL) │     │  (Chunking)     │     │  (Azure)    │
└─────────────┘     └─────────────────┘     └──────┬──────┘
                                                   │
                                                   ▼
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│   Answer    │◀────│   LLM (GPT-4)   │◀────│ Vector Store│
│             │     │                 │     │  (Chroma)   │
└─────────────┘     └────────┬────────┘     └──────┬──────┘
                             │                     │
                             │    ┌────────────┐   │
                             └────│  Retriever │◀──┘
                                  │  (Top-K)   │
                                  └─────┬──────┘
                                        │
                              ┌─────────▼─────────┐
                              │   User Question   │
                              └───────────────────┘
```

## Prerequisites

- **Python 3.10+**
- **Azure Subscription** with:
  - Azure OpenAI Service deployed
  - A chat model deployment (e.g., `gpt-4`, `gpt-35-turbo`)
  - An embedding model deployment (e.g., `text-embedding-ada-002`)
- **Azure CLI** installed and logged in (`az login`)

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/aganiezgoda/langchainex_rag.git
cd langchainex_rag
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

Edit `.env` with your Azure OpenAI settings (see [.env.example](.env.example) for details).

### 5. Authenticate with Azure

```bash
az login
```

The script uses `DefaultAzureCredential`, which supports:
- Azure CLI authentication (recommended for development)
- Managed Identity (for production on Azure)
- Visual Studio Code authentication
- Environment variables

## Usage

### Run the tutorial

```bash
python langchain_rag_tutorial.py
```

### Expected output

```
============================================================
🚀 RAG Tutorial with LangChain & Azure OpenAI
============================================================
📄 Loading document from: https://lilianweng.github.io/posts/2023-06-23-agent/
   ✅ Loaded 1 document(s)
   📏 Total characters: 43,131

✂️  Splitting documents into chunks...
   Chunk size: 1000, Overlap: 200
   ✅ Created 66 chunks

🔢 Creating embeddings and vector store...
   ✅ Vector store created with 66 vectors

🔍 Creating retriever (k=4 documents)...

⛓️  Building RAG chain...
   ✅ RAG chain ready!

============================================================
💬 Ready to answer questions!
============================================================

❓ Question: What is Task Decomposition?
------------------------------------------------------------

🤖 Answer: Task Decomposition is breaking down complex tasks into smaller, 
manageable steps. It can be done through prompting techniques like Chain of 
Thought or Tree of Thoughts, or by using task-specific instructions.

📚 Retrieved Sources:
   1. Task Decomposition is a technique used to break down complex tasks...
   2. Chain of Thought (CoT) prompting is a technique that encourages...
```

## Code Structure

| File | Description |
|------|-------------|
| `langchain_rag_tutorial.py` | Main tutorial script with detailed comments |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for environment variables |
| `.gitignore` | Git ignore rules (excludes `.env`, `.venv`) |

## Key Concepts Explained

### 1. Document Loading
The script loads a blog post about AI agents using `WebBaseLoader`. BeautifulSoup filters out navigation and focuses on the article content.

### 2. Text Splitting (Chunking)
`RecursiveCharacterTextSplitter` breaks the document into ~1000-character chunks with 200-character overlap. This ensures:
- Chunks fit within embedding model limits
- Context is preserved across chunk boundaries

### 3. Embeddings
Azure OpenAI's `text-embedding-ada-002` model converts text to 1536-dimensional vectors. Similar texts produce similar vectors, enabling semantic search.

### 4. Vector Store
ChromaDB stores vectors in-memory for fast similarity search. For production, consider Azure AI Search, Pinecone, or Weaviate.

### 5. Retrieval
The retriever finds the top-K (default: 4) most similar chunks to the user's question using cosine similarity.

### 6. Generation
GPT-4 receives the question + retrieved context and generates a grounded answer.

## Customization

Edit these constants in `langchain_rag_tutorial.py` to experiment:

```python
CHUNK_SIZE = 1000          # Try 500 for finer granularity
CHUNK_OVERLAP = 200        # Increase if context is getting cut off
NUM_DOCS_TO_RETRIEVE = 4   # More docs = more context, but may confuse LLM
BLOG_URL = "..."           # Try your own URL!
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `DefaultAzureCredentialError` | Run `az login` to authenticate |
| `Resource not found` | Check your `AZURE_OPENAI_ENDPOINT` and deployment names |
| `Rate limit exceeded` | Wait and retry, or increase your Azure quota |
| `chromadb` errors | Ensure you have `chromadb` in requirements.txt |

## Further Learning

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [Azure OpenAI Documentation](https://learn.microsoft.com/azure/ai-services/openai/)
- [Chroma Documentation](https://docs.trychroma.com/)

## License

MIT License - feel free to use this for teaching and learning!
