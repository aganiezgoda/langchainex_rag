"""
RAG (Retrieval Augmented Generation) Tutorial with LangChain & Azure OpenAI
============================================================================

This script demonstrates how to build a RAG pipeline that:
1. Loads a document from the web
2. Splits it into manageable chunks
3. Creates embeddings and stores them in a vector database
4. Retrieves relevant chunks based on a user query
5. Generates an answer using an LLM with the retrieved context

Reference: https://python.langchain.com/docs/tutorials/rag/
"""

# =============================================================================
# STEP 1: IMPORTS
# =============================================================================
import os
from dotenv import load_dotenv

# LangChain core components
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate

# LangChain integrations
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Azure authentication
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# For parsing HTML
import bs4


# =============================================================================
# STEP 2: CONFIGURATION
# =============================================================================
load_dotenv()

# Azure OpenAI settings from environment variables
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_EMBED = os.getenv("AZURE_DEPLOYMENT_EMBED")
AZURE_DEPLOYMENT_LLM = os.getenv("AZURE_DEPLOYMENT_LLM")

# RAG configuration parameters - easy to experiment with!
CHUNK_SIZE = 1000          # Size of each text chunk in characters
CHUNK_OVERLAP = 200        # Overlap between chunks to preserve context
NUM_DOCS_TO_RETRIEVE = 4   # Number of relevant documents to retrieve

# Source document URL
BLOG_URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"


def setup_azure_credentials():
    """
    Set up Azure authentication using DefaultAzureCredential.
    This supports multiple auth methods: managed identity, VS Code, Azure CLI, etc.
    """
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, 
        "https://cognitiveservices.azure.com/.default"
    )
    return token_provider


# =============================================================================
# STEP 3: DOCUMENT LOADING
# =============================================================================
def load_document(url: str) -> list:
    """
    Load a document from a web URL.
    
    We use BeautifulSoup to parse only the relevant content (post body),
    avoiding navigation menus, footers, etc.
    
    Args:
        url: The URL of the document to load
        
    Returns:
        List of Document objects
    """
    print(f"📄 Loading document from: {url}")
    
    # SoupStrainer filters HTML to only include specific elements
    bs_strainer = bs4.SoupStrainer(
        class_=("post-content", "post-title", "post-header")
    )
    
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs={"parse_only": bs_strainer},
    )
    
    docs = loader.load()
    print(f"   ✅ Loaded {len(docs)} document(s)")
    print(f"   📏 Total characters: {sum(len(d.page_content) for d in docs):,}")
    
    return docs


# =============================================================================
# STEP 4: TEXT SPLITTING (CHUNKING)
# =============================================================================
def split_documents(docs: list, chunk_size: int, chunk_overlap: int) -> list:
    """
    Split documents into smaller chunks for embedding.
    
    Why chunk?
    - LLMs have context limits
    - Smaller chunks = more precise retrieval
    - Overlap preserves context at chunk boundaries
    
    Args:
        docs: List of documents to split
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of overlapping characters between chunks
        
    Returns:
        List of Document chunks
    """
    print(f"\n✂️  Splitting documents into chunks...")
    print(f"   Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # These separators are tried in order - splits on paragraphs first
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    splits = text_splitter.split_documents(docs)
    print(f"   ✅ Created {len(splits)} chunks")
    
    # Show a sample chunk for educational purposes
    if splits:
        print(f"\n   📝 Sample chunk (first 200 chars):")
        print(f"   '{splits[0].page_content[:200]}...'")
    
    return splits


# =============================================================================
# STEP 5: VECTOR STORE CREATION
# =============================================================================
def create_vectorstore(splits: list, token_provider) -> Chroma:
    """
    Create a vector store from document chunks.
    
    Process:
    1. Each chunk is sent to the embedding model
    2. The model returns a vector (list of numbers) representing the text
    3. Vectors are stored in ChromaDB for fast similarity search
    
    Args:
        splits: List of document chunks
        token_provider: Azure token provider for authentication
        
    Returns:
        Chroma vector store instance
    """
    print(f"\n🔢 Creating embeddings and vector store...")
    
    # Initialize the embedding model
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        azure_deployment=AZURE_DEPLOYMENT_EMBED,
        openai_api_version=OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_ad_token_provider=token_provider
    )
    
    # Create vector store from documents
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )
    
    print(f"   ✅ Vector store created with {len(splits)} vectors")
    
    return vectorstore


# =============================================================================
# STEP 6: RETRIEVER SETUP
# =============================================================================
def create_retriever(vectorstore: Chroma, k: int = 4):
    """
    Create a retriever from the vector store.
    
    The retriever finds the k most similar documents to a query
    using cosine similarity between embedding vectors.
    
    Args:
        vectorstore: The Chroma vector store
        k: Number of documents to retrieve
        
    Returns:
        A retriever instance
    """
    print(f"\n🔍 Creating retriever (k={k} documents)...")
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    return retriever


# =============================================================================
# STEP 7: RAG CHAIN CONSTRUCTION
# =============================================================================
def create_rag_chain(retriever, token_provider):
    """
    Build the RAG chain that combines retrieval and generation.
    
    Chain flow:
    1. User question comes in
    2. Retriever finds relevant documents
    3. Documents are formatted into context
    4. Prompt template combines question + context
    5. LLM generates the answer
    6. Output parser extracts the text
    
    Args:
        retriever: The document retriever
        token_provider: Azure token provider
        
    Returns:
        The complete RAG chain
    """
    print(f"\n⛓️  Building RAG chain...")
    
    # Initialize the LLM
    llm = AzureChatOpenAI(
        openai_api_version=OPENAI_API_VERSION,
        azure_deployment=AZURE_DEPLOYMENT_LLM,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_ad_token_provider=token_provider,
        temperature=0  # Deterministic output for reproducibility
    )
    
    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("human", """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:""")
    ])
    
    # Helper function to format retrieved documents
    def format_docs(docs):
        """Join document contents with double newlines."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Build the chain that processes documents
    rag_chain_from_docs = (
        # Assign formatted context from the retrieved documents
        RunnablePassthrough.assign(
            context=lambda x: format_docs(x["context"])
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Build the full chain with source tracking
    rag_chain_with_source = RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough()
    }).assign(answer=rag_chain_from_docs)
    
    print("   ✅ RAG chain ready!")
    
    return rag_chain_with_source


# =============================================================================
# STEP 8: QUERY EXECUTION
# =============================================================================
def ask_question(rag_chain, question: str, show_sources: bool = True) -> dict:
    """
    Ask a question using the RAG chain.
    
    Args:
        rag_chain: The RAG chain to use
        question: The question to ask
        show_sources: Whether to display retrieved sources
        
    Returns:
        Dictionary with 'answer' and 'context' keys
    """
    print(f"\n❓ Question: {question}")
    print("-" * 60)
    
    # Stream the response for real-time feedback
    output = {}
    curr_key = None
    
    print("\n🤖 Answer: ", end="", flush=True)
    
    for chunk in rag_chain.stream(question):
        for key in chunk:
            if key not in output:
                output[key] = chunk[key]
            else:
                output[key] += chunk[key]
            
            # Only print the answer part (not context)
            if key == "answer":
                print(chunk[key], end="", flush=True)
    
    print("\n")
    
    # Show sources if requested
    if show_sources and "context" in output:
        print("📚 Retrieved Sources:")
        for i, doc in enumerate(output["context"], 1):
            preview = doc.page_content[:150].replace("\n", " ")
            print(f"   {i}. {preview}...")
    
    return output


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """
    Main function demonstrating the complete RAG pipeline.
    """
    print("=" * 60)
    print("🚀 RAG Tutorial with LangChain & Azure OpenAI")
    print("=" * 60)
    
    # Setup authentication
    token_provider = setup_azure_credentials()
    
    # Step 1: Load the document
    docs = load_document(BLOG_URL)
    
    # Step 2: Split into chunks
    splits = split_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    
    # Step 3: Create vector store with embeddings
    vectorstore = create_vectorstore(splits, token_provider)
    
    # Step 4: Create retriever
    retriever = create_retriever(vectorstore, k=NUM_DOCS_TO_RETRIEVE)
    
    # Step 5: Build RAG chain
    rag_chain = create_rag_chain(retriever, token_provider)
    
    # Step 6: Ask questions!
    print("\n" + "=" * 60)
    print("💬 Ready to answer questions!")
    print("=" * 60)
    
    # Example questions
    questions = [
        "What is Task Decomposition?",
        "What are the different types of memory in an AI agent?",
    ]
    
    for question in questions:
        ask_question(rag_chain, question, show_sources=True)
        print("\n" + "-" * 60)
    
    # Cleanup
    print("\n🧹 Cleaning up vector store...")
    vectorstore.delete_collection()
    print("✅ Done!")


if __name__ == "__main__":
    main()