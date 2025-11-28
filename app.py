# app.py
# RAG + tools + memory agent with LangChain + LangSmith + Gradio for Render

import os
import json
from datetime import datetime

import torch
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
import gradio as gr


# ========= Environment variables =========
# On Render, set these in the dashboard:
# - OPENAI_API_KEY
# - PINECONE_API_KEY
# - LANGCHAIN_API_KEY
# - LANGCHAIN_TRACING_V2=true
# - LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
# - LANGCHAIN_PROJECT=memory-and-tools-rag-agent

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY env var is not set.")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY env var is not set.")

# ========= RAG config =========

INDEX_NAME = "youtube-qa-index"
TOP_K = 5  # how many passages to retrieve from Pinecone

device = "cuda" if torch.cuda.is_available() else "cpu"
retriever = SentenceTransformer(
    "flax-sentence-embeddings/all_datasets_v3_mpnet-base",
    device=device,
)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
print(f"Connected to Pinecone index: {INDEX_NAME}")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Global in-memory conversation history
memory = []


# ========= RAG helpers =========

def retrieve_pinecone_context(query: str, top_k: int = TOP_K):
    """Query Pinecone with an embedding of the user query."""
    xq = retriever.encode(query).tolist()
    res = index.query(vector=xq, top_k=top_k, include_metadata=True)
    return res


def context_string_from_matches(matches):
    """Build a single context string from Pinecone matches."""
    parts = []
    for m in matches:
        meta = m["metadata"]
        passage = meta.get("text") or meta.get("passage_text") or ""
        if passage:
            parts.append(passage)
    return "\n\n".join(parts)


# ========= Tools =========

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Example: '2 + 2 * 5' or '10 / 3'
    """
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def word_count(text: str) -> str:
    """Count the number of words in a given text."""
    count = len(text.split())
    return f"Word count: {count}"


@tool
def convert_case(text: str, case_type: str) -> str:
    """
    Convert text to uppercase, lowercase, or title case.
    case_type options: 'upper', 'lower', 'title'
    """
    if case_type == "upper":
        return text.upper()
    elif case_type == "lower":
        return text.lower()
    elif case_type == "title":
        return text.title()
    else:
        return (
            f"Error: Unknown case type '{case_type}'. "
            "Use 'upper', 'lower', or 'title'."
        )


tools = [calculator, get_current_time, word_count, convert_case]
print(
    f"✅ Loaded {len(tools)} tools: calculator, get_current_time, "
    f"word_count, convert_case"
)

llm_with_tools = llm.bind_tools(tools)


# ========= Runnable chain (agent) =========

def _build_messages
