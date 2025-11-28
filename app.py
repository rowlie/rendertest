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

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY env var is not set.")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_
