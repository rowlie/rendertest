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

def _build_messages(inputs: dict):
    """
    Inputs: {"user_message": str}
    Output: {"messages": List[BaseMessage], "rag_context": str}
    """
    user_message = inputs["user_message"]

    # RAG: retrieve context from Pinecone
    pinecone_result = retrieve_pinecone_context(user_message)
    context = context_string_from_matches(pinecone_result.get("matches", []))

    # Start from global memory
    messages = list(memory)
    messages.append(HumanMessage(content=user_message))

    if context:
        messages.append(
            HumanMessage(
                content=f"📚 Relevant context from knowledge base:\n{context}"
            )
        )

    return {
        "messages": messages,
        "rag_context": context,
    }


build_messages = RunnableLambda(_build_messages)


def _first_llm_call(state: dict):
    """Call tool-enabled LLM to decide whether to use tools."""
    messages = state["messages"]
    first_response = llm_with_tools.invoke(messages)
    return {
        **state,
        "first_response": first_response,
    }


first_llm_call = RunnableLambda(_first_llm_call)


def _run_tools_if_needed(state: dict):
    """Execute any tools requested by the first LLM call."""
    first_response = state["first_response"]
    messages = state["messages"]

    tool_calls = getattr(first_response, "tool_calls", None)
    if not tool_calls and hasattr(first_response, "additional_kwargs"):
        tool_calls = first_response.additional_kwargs.get("tool_calls")

    if not tool_calls:
        # No tools requested, just pass messages through
        return {
            **state,
            "messages_with_tools": messages,
        }

    tool_results_messages = []
    for call in tool_calls:
        tool_name = call.get("name") or call.get("function", {}).get("name")
        raw_args = (
            call.get("args")
            or call.get("arguments")
            or call.get("function", {}).get("arguments", {})
        )

        if isinstance(raw_args, str):
            try:
                tool_args = json.loads(raw_args)
            except Exception:
                tool_args = {}
        else:
            tool_args = raw_args or {}

        tool_id = call.get("id", "tool_call")

        matching = [t for t in tools if t.name == tool_name]
        if not matching:
            result_text = f"Tool '{tool_name}' not found."
        else:
            try:
                result_text = matching[0].invoke(tool_args)
            except Exception as e:
                result_text = f"Error in tool '{tool_name}': {e}"

        tool_results_messages.append(
            ToolMessage(
                content=str(result_text),
                tool_call_id=tool_id,
            )
        )

    messages_with_tools = messages + [first_response] + tool_results_messages

    return {
        **state,
        "messages_with_tools": messages_with_tools,
    }


run_tools_if_needed = RunnableLambda(_run_tools_if_needed)


def _final_llm_call(state: dict):
    """Call plain LLM for final answer, given tool outputs (if any)."""
    messages_with_tools = state["messages_with_tools"]
    final_response = llm.invoke(messages_with_tools)
    return {
        "final_response": final_response,
        "rag_context": state.get("rag_context", ""),
    }


final_llm_call = RunnableLambda(_final_llm_call)


rag_agent_chain = (
    RunnableLambda(lambda user_message: {"user_message": user_message})
    | build_messages
    | first_llm_call
    | run_tools_if_needed
    | final_llm_call
)


def chat_with_rag_and_tools(user_message: str) -> str:
    """
    Wrapper around the Runnable chain:
    - Calls rag_agent_chain(user_message)
    - Updates global memory with this turn
    - Returns final_response.content
    """
    result = rag_agent_chain.invoke(user_message)
    final_response = result["final_response"]
    rag_context = result.get("rag_context", "")

    # Update memory (conversation history)
    memory.append(HumanMessage(content=user_message))
    if rag_context:
        memory.append(
            HumanMessage(
                content=f"📚 Relevant context from knowledge base:\n{rag_context}"
            )
        )
    memory.append(AIMessage(content=final_response.content))

    return final_response.content


# ========= Gradio app (for Render) =========

def gradio_chat(user_message, chat_history):
    """Gradio callback: uses LangChain agent and returns updated chat history."""
    if not user_message:
        return "", chat_history

    answer = chat_with_rag_and_tools(user_message)
    chat_history = (chat_history or []) + [[user_message, answer]]
    return "", chat_history


with gr.Blocks() as demo:
    gr.Markdown("## YouTube RAG Chatbot with Tools + LangSmith")
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(label="Ask a question about your YouTube videos")
    clear = gr.Button("Clear")

    msg.submit(gradio_chat, inputs=[msg, chatbot], outputs=[msg, chatbot])
    clear.click(lambda: ("", []), outputs=[msg, chatbot])


# This `app` object is what Render will serve.
app = demo


if __name__ == "__main__":
    # Explicit Gradio server so Render detects the bound port
    port = int(os.environ.get("PORT", 10000))
    server = gr.Server(app, server_name="0.0.0.0", server_port=port)
    server.launch()
