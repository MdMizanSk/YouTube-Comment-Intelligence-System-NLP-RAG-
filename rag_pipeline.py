"""
rag_pipeline.py
---------------
Core RAG (Retrieval-Augmented Generation) pipeline.
Responsibilities:
  - Convert raw comments into LangChain Documents
  - Chunk documents with RecursiveCharacterTextSplitter
  - Embed chunks with OpenAIEmbeddings and store in FAISS
  - Build a retriever (k=5)
  - Define a ChatPromptTemplate for comment analysis
  - Expose an `answer_query` function that runs the full RAG chain
"""

import os
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------------------------
# 1. Document Preparation
# ---------------------------------------------------------------------------

def build_documents(comments: list[str]) -> list[Document]:
    """
    Convert a list of raw comment strings into LangChain Document objects.

    Each comment becomes one Document. Metadata stores the original index
    so we can trace retrieved chunks back to source comments if needed.

    Args:
        comments (list[str]): Raw YouTube comment strings

    Returns:
        list[Document]: LangChain Document objects
    """
    return [
        Document(page_content=comment, metadata={"index": i})
        for i, comment in enumerate(comments)
    ]


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into smaller chunks for embedding.

    Uses RecursiveCharacterTextSplitter with:
      - chunk_size   = 300  (characters per chunk)
      - chunk_overlap = 50  (overlap to preserve context across chunk boundaries)

    Args:
        documents (list[Document]): Full LangChain Documents

    Returns:
        list[Document]: Chunked Documents ready for embedding
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
    )
    return splitter.split_documents(documents)


# ---------------------------------------------------------------------------
# 2. Vector Store + Retriever
# ---------------------------------------------------------------------------

def build_vectorstore(chunks: list[Document]) -> FAISS:
    """
    Embed document chunks with OpenAIEmbeddings and store in a FAISS index.

    Args:
        chunks (list[Document]): Chunked documents

    Returns:
        FAISS: In-memory FAISS vector store
    """
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def build_retriever(vectorstore: FAISS, k: int = 5):
    """
    Create a retriever from the FAISS vector store.

    Args:
        vectorstore (FAISS): Populated FAISS index
        k           (int)  : Number of top documents to retrieve per query

    Returns:
        VectorStoreRetriever: LangChain retriever
    """
    return vectorstore.as_retriever(search_kwargs={"k": k})


# ---------------------------------------------------------------------------
# 3. Prompt Engineering
# ---------------------------------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a YouTube comment analyst. "
                "Provide actionable insights based on viewer comments. "
                "Be concise, direct, and always ground your answers in the provided context."
            ),
        ),
        (
            "human",
            (
                "Question: {question}\n\n"
                "Relevant comments (context):\n{context}\n\n"
                "Using only the context above, answer the question."
            ),
        ),
        (
            "ai",
            (
                "I will analyze the provided comments carefully and give a "
                "structured, concise answer based strictly on the context given."
            ),
        ),
    ]
)


# ---------------------------------------------------------------------------
# 4. RAG Chain
# ---------------------------------------------------------------------------

def build_llm() -> ChatOpenAI:
    """
    Instantiate the ChatOpenAI model used for RAG answering.

    Returns:
        ChatOpenAI: LLM instance (gpt-4o-mini, temperature=0 for determinism)
    """
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )


def answer_query(query: str, retriever, llm: ChatOpenAI) -> str:
    """
    Full RAG pipeline: retrieve → inject context → generate answer.

    Steps:
      1. Use the retriever to fetch k most relevant comment chunks
      2. Concatenate chunks into a single context string
      3. Inject question + context into the RAG prompt
      4. Run through the LLM
      5. Parse output with StrOutputParser

    Args:
        query     (str)           : User's natural language question
        retriever                 : FAISS-backed LangChain retriever
        llm       (ChatOpenAI)    : Instantiated LLM

    Returns:
        str: LLM-generated answer grounded in retrieved comments
    """
    # Step 1: Retrieve relevant documents
    relevant_docs = retriever.invoke(query)

    # Step 2: Build context string from retrieved chunks
    context = "\n\n".join(
        [f"Comment {i+1}: {doc.page_content}" for i, doc in enumerate(relevant_docs)]
    )

    # Step 3 + 4: Build and run the chain
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"question": query, "context": context})

    return answer


# ---------------------------------------------------------------------------
# 5. One-shot pipeline builder (convenience)
# ---------------------------------------------------------------------------

def build_rag_pipeline(comments: list[str]):
    """
    End-to-end convenience function: from raw comments to retriever + llm.

    Args:
        comments (list[str]): Raw YouTube comment strings

    Returns:
        tuple: (retriever, llm) ready to be passed into answer_query()
    """
    docs = build_documents(comments)
    chunks = chunk_documents(docs)
    vectorstore = build_vectorstore(chunks)
    retriever = build_retriever(vectorstore, k=5)
    llm = build_llm()
    return retriever, llm
