# app.py
# RAG + tools + memory agent with LangChain + LangSmith + Gradio for Render

import os
from datetime import datetime
import json

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
    raise ValueError("OPENAI_API_KEY not set.")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set.")


# ========= RAG config =========

INDEX_NAME = "youtube-qa-index"
TOP_K = 5

device = "cuda" if torch.cuda.is_available() else "cpu"

retriever = SentenceTransformer(
    "flax-sentence-embeddings/all_datasets_v3_mpnet-base",
    device=device,
)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
print(f"Connected to Pinecone index: {INDEX_NAME}")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Store only raw text, not LangChain objects
memory = []


# ========= RAG helpers =========

def retrieve_pinecone_context(query: str, top_k: int = TOP_K):
    vec = retriever.encode(query).tolist()
    result = index.query(vector=vec, top_k=top_k, include_metadata=True)
    return result.matches or []


def context_string_from_matches(matches):
    out = []
    for m in matches:
        md = m.metadata or {}
        passage = md.get("text") or md.get("passage_text")
        if passage:
            out.append(passage)
    return "\n\n".join(out)


# ========= Tools =========

@tool
def calculator(expression: str) -> str:
    """Evaluate math expressions."""
    try:
        return f"Result: {eval(expression)}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def word_count(text: str) -> str:
    return f"Word count: {len(text.split())}"


@tool
def convert_case(text: str, case_type: str) -> str:
    if case_type == "upper":
        return text.upper()
    if case_type == "lower":
        return text.lower()
    if case_type == "title":
        return text.title()
    return "Error: use 'upper', 'lower', or 'title'."


tools = [calculator, get_current_time, word_count, convert_case]
llm_with_tools = llm.bind_tools(tools)

print("Tools loaded:", [t.name for t in tools])


# ========= Agent chain =========

def _build_messages(inputs: dict):
    user_msg = inputs["user_message"]

    matches = retrieve_pinecone_context(user_msg)
    context = context_string_from_matches(matches)

    messages = []
    for m in memory:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))

    # Add user
    messages.append(HumanMessage(content=user_msg))

    # Inject RAG context
    if context:
        messages.append(
            HumanMessage(
                content=f"📚 Relevant context from knowledge base:\n{context}"
            )
        )

    return {"messages": messages, "rag_context": context}


build_messages = RunnableLambda(_build_messages)


def _first_llm_call(state: dict):
    resp = llm_with_tools.invoke(state["messages"])
    return {**state, "first_response": resp}


first_llm_call = RunnableLambda(_first_llm_call)


def _run_tools_if_needed(state: dict):
    resp = state["first_response"]
    tool_calls = getattr(resp, "tool_calls", None)

    messages = state["messages"]

    if not tool_calls:
        return {"messages_with_tools": messages + [resp], **state}

    tool_messages = []

    for call in tool_calls:
        tool_name = call.get("name")
        tool_args = call.get("args", {})
        tool_id = call.get("id", "tool_call")

        match = next((t for t in tools if t.name == tool_name), None)
        if not match:
            result = f"Tool '{tool_name}' not found."
        else:
            try:
                result = match.run(tool_args)
            except Exception as e:
                result = f"Error: {str(e)}"

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_id)
        )

    return {
        **state,
        "messages_with_tools": messages + [resp] + tool_messages
    }


run_tools_if_needed = RunnableLambda(_run_tools_if_needed)


def _final_llm_call(state: dict):
    answer = llm.invoke(state["messages_with_tools"])
    return {"final_response": answer, "rag_context": state.get("rag_context")}


final_llm_call = RunnableLambda(_final_llm_call)


rag_agent_chain = (
    RunnableLambda(lambda x: {"user_message": x})
    | build_messages
    | first_llm_call
    | run_tools_if_needed
    | final_llm_call
)


def chat_with_rag_and_tools(user_message: str) -> str:
    result = rag_agent_chain.invoke(user_message)

    # Store into simple memory
    memory.append({"role": "user", "content": user_message})
    memory.append({"role": "assistant", "content": result["final_response"].content})

    return result["final_response"].content


# ========= Gradio App =========

def gradio_chat(user_message, chat_history):
    if not user_message:
        return "", chat_history

    reply = chat_with_rag_and_tools(user_message)
    chat_history = (chat_history or []) + [[user_message, reply]]
    return "", chat_history


with gr.Blocks() as demo:
    gr.Markdown("## YouTube RAG Chatbot with Tools + LangSmith")
    chatbot = gr.Chatbot(height=400, show_label=False)
    msg = gr.Textbox(label="Ask anything")
    clear = gr.Button("Clear")

    msg.submit(gradio_chat, inputs=[msg, chatbot], outputs=[msg, chatbot])
    clear.click(lambda: ("", []), outputs=[msg, chatbot])


# ========= REQUIRED FOR RENDER =========

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    demo.queue()

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_api=False
    )
