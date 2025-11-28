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
    raise ValueError("PINECONE_API_KEY env var is not set.")


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

openai_client = OpenAI(api_key=OPENAI_API_KEY)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

memory = []  # global conversation memory


# ========= RAG helpers =========

def retrieve_pinecone_context(query: str, top_k: int = TOP_K):
    xq = retriever.encode(query).tolist()
    res = index.query(vector=xq, top_k=top_k, include_metadata=True)
    return res


def context_string_from_matches(matches):
    parts = []
    for m in matches:
        passage = m["metadata"].get("text") or m["metadata"].get("passage_text") or ""
        if passage:
            parts.append(passage)
    return "\n\n".join(parts)


# ========= Tools =========

@tool
def calculator(expression: str) -> str:
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
print("Loaded tools:", tools)

llm_with_tools = llm.bind_tools(tools)


# ========= Agent chain =========

def _build_messages(inputs: dict):
    user_message = inputs["user_message"]

    pinecone_result = retrieve_pinecone_context(user_message)
    context = context_string_from_matches(pinecone_result.get("matches", []))

    messages = list(memory)
    messages.append(HumanMessage(content=user_message))

    if context:
        messages.append(
            HumanMessage(
                content=f"📚 Relevant context from knowledge base:\n{context}"
            )
        )

    return {"messages": messages, "rag_context": context}


build_messages = RunnableLambda(_build_messages)


def _first_llm_call(state: dict):
    response = llm_with_tools.invoke(state["messages"])
    return {**state, "first_response": response}


first_llm_call = RunnableLambda(_first_llm_call)


def _run_tools_if_needed(state: dict):
    first_response = state["first_response"]
    messages = state["messages"]

    tool_calls = getattr(first_response, "tool_calls", None)
    if not tool_calls and hasattr(first_response, "additional_kwargs"):
        tool_calls = first_response.additional_kwargs.get("tool_calls")

    if not tool_calls:
        return {"messages_with_tools": messages, **state}

    tool_messages = []

    for call in tool_calls:
        tool_name = call.get("name")
        args = call.get("args") or {}
        tool_id = call.get("id", "tool_call")

        matching = [t for t in tools if t.name == tool_name]
        if not matching:
            result = f"Tool '{tool_name}' not found."
        else:
            try:
                result = matching[0].invoke(args)
            except Exception as e:
                result = f"Error: {e}"

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_id)
        )

    return {
        **state,
        "messages_with_tools": messages + [first_response] + tool_messages,
    }


run_tools_if_needed = RunnableLambda(_run_tools_if_needed)


def _final_llm_call(state: dict):
    final_response = llm.invoke(state["messages_with_tools"])
    return {"final_response": final_response, "rag_context": state.get("rag_context", "")}


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
    final_resp = result["final_response"]
    rag_context = result.get("rag_context", "")

    memory.append(HumanMessage(content=user_message))
    if rag_context:
        memory.append(HumanMessage(content=f"📚 Relevant context:\n{rag_context}"))
    memory.append(AIMessage(content=final_resp.content))

    return final_resp.content


# ========= Gradio App =========

def gradio_chat(user_message, chat_history):
    if not user_message:
        return "", chat_history
    answer = chat_with_rag_and_tools(user_message)
    return "", (chat_history or []) + [[user_message, answer]]


with gr.Blocks() as demo:
    gr.Markdown("## YouTube RAG Chatbot with Tools + LangSmith")
    chatbot = gr.Chatbot(height=400, show_label=False)
    msg = gr.Textbox(label="Ask")
    clear = gr.Button("Clear")

    msg.submit(gradio_chat, inputs=[msg, chatbot], outputs=[msg, chatbot])
    clear.click(lambda: ("", []), outputs=[msg, chatbot])


# ========= REQUIRED FOR RENDER =========

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    demo.queue()  # VERY IMPORTANT FOR RENDER + GRADIO 6

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_api=False
    )
