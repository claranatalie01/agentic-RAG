import os
import logging
import json
import asyncio
import urllib.parse
import torch
import torch.nn.functional as F
import re
from typing import Optional, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
from gliner2 import GLiNER2
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from .state import LibraryBotState
from .utils import get_current_datetime

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MCP_SERVER_URL = "http://localhost:8000/mcp"

# ----------------------------------------------------------------------
# Helper mappings (keyword‑based intent routing)
# ----------------------------------------------------------------------
LIBRARY_NAME_TO_CODE = {
    "central": "HKCL",
    "hong kong central": "HKCL",
    "shatin": "STPL",
    "sha tin": "STPL",
    "tsim sha tsui": "TSTPL",
    "kowloon": "KLNPL",
}

KEYWORD_TO_TOOL = {
    "workstation|computer|pc|availability": "get_live_workstation_availability",
    "address|phone|email|details|about library": "get_library_details",
    "opening hours|open|hours": "get_library_opening_hours",
    "how many libraries|library count|total libraries": "get_library_count",
    "search for|find book|catalog|title|author": "search_library_catalog",
    "available|in stock|check if": "check_book_availability",
    "nearby|near me|close to|libraries in|district": "find_nearby_libraries",
}

# ----------------------------------------------------------------------
# Database and embeddings (Qwen3-Embedding-0.6B)
# ----------------------------------------------------------------------
DB_PASSWORD = os.getenv("DB_PASSWORD")
if not DB_PASSWORD:
    raise ValueError("DB_PASSWORD not set in .env")
encoded_password = urllib.parse.quote(DB_PASSWORD, safe='')
CONNECTION_STRING = f"postgresql+psycopg2://postgres:{encoded_password}@localhost:5433/hkpl_vector_db"
engine = create_engine(CONNECTION_STRING, pool_size=10, max_overflow=20)

# Load Qwen3-Embedding-0.6B
embed_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
embed_model.to("cpu")   # or "cuda"

class Qwen3Embeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        # For indexing documents (we use this during ingestion)
        return self.model.encode(texts, prompt_name="document", normalize_embeddings=True).tolist()

    def embed_query(self, text):
        # For queries at retrieval time
        return self.model.encode(text, prompt_name="query", normalize_embeddings=True).tolist()

embeddings = Qwen3Embeddings(embed_model)

executor = ThreadPoolExecutor(max_workers=10)

async def run_db_query(query_text: str, params: dict):
    loop = asyncio.get_running_loop()
    with engine.connect() as conn:
        result = await loop.run_in_executor(
            executor,
            lambda: conn.execute(text(query_text), params).fetchall()
        )
    return result


# ----------------------------------------------------------------------
# Local LLMs (glm)
# ----------------------------------------------------------------------
llm_deterministic = ChatOllama(
    model="glm-4.7-flash",
    temperature=0,
    num_predict=2048,
    top_p=0.95
)

llm_chat = ChatOllama(
    model="glm-4.7-flash",
    temperature=0.7,
    num_predict=2048,
    top_p=0.95
)

# ----------------------------------------------------------------------
# Reranker (transformers‑based)
# ----------------------------------------------------------------------
torch.set_num_threads(2)           # Reduce CPU contention
torch.backends.mps.is_available = lambda: False   # Disable MPS detection
torch.backends.mps.is_built = lambda: False
reranker_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", trust_remote_code=True)
reranker_model = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen3-Reranker-0.6B")
reranker_model.eval()
reranker_device = torch.device("cpu")
reranker_model.to(reranker_device)
# ----------------------------------------------------------------------
# Nodes
# ----------------------------------------------------------------------
async def voice_to_text_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Voice-to-Text Conversion (mock)")
    transcribed = "請問沙田圖書館幾點關門？"
    return {"messages": [HumanMessage(content=transcribed)]}

# ----------------------------------------------------------------------
# GLiGuard safety classifier
# ----------------------------------------------------------------------
safety_model = None

def get_safety_model():
    global safety_model
    if safety_model is None:
        safety_model = GLiNER2.from_pretrained("fastino/gliguard-LLMGuardrails-300M")
        safety_model.to("cpu")
    return safety_model

async def safety_and_intent_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Safety Classifier (GLiGuard)")
    user_input = state["messages"][-1].content
    model = get_safety_model()

    toxicity_labels = [
        "violence_and_weapons", "non_violent_crime", "sexual_content",
        "hate_and_discrimination", "self_harm_and_suicide", "pii_exposure",
        "misinformation", "copyright_violation", "child_safety",
        "political_manipulation", "unethical_conduct", "regulated_advice",
        "privacy_violation", "other", "benign",
    ]
    toxicity_task = {
        "labels": toxicity_labels,
        "multi_label": True,
        "cls_threshold": 0.4,
    }
    jailbreak_labels = [
        "prompt_injection", "jailbreak_attempt", "policy_evasion",
        "instruction_override", "system_prompt_exfiltration", "data_exfiltration",
        "roleplay_bypass", "hypothetical_bypass", "obfuscated_attack",
        "multi_step_attack", "social_engineering", "benign",
    ]
    jailbreak_task = {
        "labels": jailbreak_labels,
        "multi_label": True,
        "cls_threshold": 0.4,
    }
    schema = {
        "prompt_safety": ["safe", "unsafe"],
        "prompt_toxicity": toxicity_task,
        "jailbreak_detection": jailbreak_task,
    }

    try:
        result = model.classify_text(user_input, schema, threshold=0.5)
        safety = result.get("prompt_safety", "safe")
        is_unsafe = (safety == "unsafe")
        detected_categories = []
        toxicity = result.get("prompt_toxicity", [])
        if isinstance(toxicity, list):
            detected_categories.extend(toxicity)
        jailbreak = result.get("jailbreak_detection", [])
        if isinstance(jailbreak, list):
            detected_categories.extend(jailbreak)
    except Exception as e:
        logger.error(f"GLiGuard classification error: {e}")
        is_unsafe = False
        detected_categories = []

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input[:500],
        "safety": "unsafe" if is_unsafe else "safe",
        "categories": detected_categories
    }

    if is_unsafe:
        with open("safety_intent_log.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        logger.warning(f"Unsafe input blocked: {user_input[:100]} | categories: {detected_categories}")
        if any(cat in ["self_harm_and_suicide", "self_harm", "suicide"] for cat in detected_categories):
            safe_msg = (
                "I'm really sorry you're feeling this way. Please know that you're not alone. "
                "If you are in distress, please reach out to the Samaritans Hong Kong 24‑hour hotline at 2896 0000, "
                "or contact a mental health professional. Your well‑being is very important."
            )
        elif "political_manipulation" in detected_categories:
            safe_msg = (
                "I'm here to help with library services, book information, and general library questions. "
                "I can't discuss political topics. Is there something about the library I can help you with?"
            )
        elif any(cat in ["prompt_injection", "jailbreak_attempt", "instruction_override"] for cat in detected_categories):
            safe_msg = (
                "I can only follow instructions related to library services. "
                "Please ask a genuine question about library hours, book searches, or library policies."
            )
        elif "violence_and_weapons" in detected_categories:
            safe_msg = (
                "I cannot provide information that promotes or glorifies violence. "
                "If you need help with library resources, I'm happy to assist."
            )
        elif "hate_and_discrimination" in detected_categories:
            safe_msg = (
                "I strive to be respectful and helpful to everyone. "
                "Please ask a library‑related question without using offensive language."
            )
        else:
            safe_msg = (
                "I'm unable to process that request. Please ask a library‑related question, such as "
                "library hours, book availability, or how to borrow materials."
            )
        return {
            "messages": [AIMessage(content=safe_msg)],
            "is_output_safe": True,
            "end_conversation": True
        }

    return {}

# ----------------------------------------------------------------------
# Intent router (keyword‑based)
# ----------------------------------------------------------------------
async def intent_router_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Intent Router (keyword‑based)")
    user_input = state["messages"][-1].content.lower()

    for pattern, tool_name in KEYWORD_TO_TOOL.items():
        if re.search(pattern, user_input):
            args = {}
            if tool_name in ("get_library_details", "get_library_opening_hours", "get_live_workstation_availability"):
                lib_code = None
                for name, code in LIBRARY_NAME_TO_CODE.items():
                    if name in user_input:
                        lib_code = code
                        break
                if lib_code:
                    args["library_code"] = lib_code
                elif tool_name != "get_live_workstation_availability":
                    args["library_code"] = "HKCL"
            elif tool_name == "search_library_catalog":
                query_clean = re.sub(r"^(search for|find book|catalog for|find|search)\s+", "", user_input)
                args["query"] = query_clean
                args["limit"] = 5
            elif tool_name == "check_book_availability":
                args["title"] = user_input
                args["author"] = None
            elif tool_name == "find_nearby_libraries":
                match = re.search(r"in (\w+(?: \w+)?) district", user_input)
                args["district"] = match.group(1) if match else "Sha Tin"
            return {
                "request_type": "mcp_tool",
                "tool_name": tool_name,
                "tool_args": args
            }

    if any(word in user_input for word in ["hello", "hi", "hey", "greeting"]):
        return {"request_type": "normal_info"}
    else:
        return {"request_type": "rag_search"}

# ----------------------------------------------------------------------
# RAG pipeline (retrieve top‑3 raw chunks)
# ----------------------------------------------------------------------
async def rag_pipeline_node(state: LibraryBotState) -> dict:
    logger.info("[Node] RAG Pipeline")
    query = state["messages"][-1].content
    loop = asyncio.get_running_loop()
    query_vector = await loop.run_in_executor(executor, embeddings.embed_query, query)
    query_vector_str = "[" + ",".join(str(x) for x in query_vector) + "]"
    rows = await run_db_query(
        "SELECT text FROM document_chunks ORDER BY embedding <=> CAST(:embedding AS vector) LIMIT 3",
        {"embedding": query_vector_str}
    )
    chunk_texts = [row[0] for row in rows]
    context = "\n\n".join(chunk_texts) if chunk_texts else "No relevant documents found."
    logger.info(f"RAG retrieved {len(chunk_texts)} chunks")
    return {
        "retrieved_chunks": chunk_texts,
        "retrieved_context": context
    }

# ----------------------------------------------------------------------
# Reranker (Qwen3-Reranker-0.6B)
# ----------------------------------------------------------------------
async def rerank_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Reranking retrieved chunks")
    query = state["messages"][-1].content
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {"retrieved_context": state.get("retrieved_context", ""), "is_relevant": False}
    
    # Truncate to at most 5 chunks for reranking (reduce memory)
    chunks = chunks[:5]
    
    # Prepare all pairs
    pairs = [[query, chunk] for chunk in chunks]
    
    # Tokenize all at once (batch)
    inputs = reranker_tokenizer(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = reranker_model(**inputs)
        logits = outputs.logits  # (batch, 2)
        probs = torch.softmax(logits, dim=1)
        scores = probs[:, 1].cpu().tolist()
    
    # Sort by relevance and take top 3
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_chunks = [chunks[i] for i in sorted_indices[:3]]
    context = "\n\n".join(top_chunks)
    return {"retrieved_context": context, "is_relevant": len(top_chunks) > 0}

# ----------------------------------------------------------------------
# Grade and rewrite (deterministic LLM)
# ----------------------------------------------------------------------
async def grade_docs_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Grade Documents")
    context = state.get("retrieved_context", "")
    question = state["messages"][-1].content
    if not context or context == "No relevant documents found.":
        return {"is_relevant": False}
    prompt = ChatPromptTemplate.from_template("""
    Determine if the provided context contains any useful information that could help answer the user's question, even if not complete. Answer "YES" or "NO".

    Context: {context}
    Question: {question}
    """)
    chain = prompt | llm_deterministic | StrOutputParser()
    try:
        result = await chain.ainvoke({"context": context, "question": question})
        is_relevant = result.strip().upper() == "YES"
    except Exception as e:
        logger.error(f"Grade error: {e}")
        is_relevant = True
    return {"is_relevant": is_relevant}

async def rewrite_query_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Rewrite Query")
    original = state["messages"][-1].content
    rewrite_count = state.get("rewrite_count", 0) + 1
    prompt = ChatPromptTemplate.from_template("""
    Rewrite the following user question to be more specific, detailed, and effective for a vector similarity search in a library knowledge base.
    Keep the original intent but improve clarity and keywords.

    Original: {original}
    Rewritten:
    """)
    chain = prompt | llm_deterministic | StrOutputParser()
    rewritten = await chain.ainvoke({"original": original})
    new_messages = state["messages"][:-1] + [HumanMessage(content=rewritten)]
    return {"messages": new_messages, "rewrite_count": rewrite_count}

# ----------------------------------------------------------------------
# Legacy API node (optional)
# ----------------------------------------------------------------------
async def utility_api_node(state: LibraryBotState) -> dict:
    logger.info("[Node] External API Call (HKPL)")
    import aiohttp
    api_url = "https://sls.hkpl.gov.hk/api/cfm-admin-service/open-api/library/selectLibraryPageInfoForPSI"
    params = {"language": "en-US", "sizePerPage": "9999"}
    timeout = aiohttp.ClientTimeout(total=5.0)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(api_url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, list) and len(data) > 0:
                        total_wkt = 0
                        for lib in data:
                            for sess in lib.get("sessionList", []):
                                for grp in sess.get("workstationGroup", []):
                                    total_wkt += grp.get("availableWktNumber", 0)
                        context = f"Live HKPL data: {len(data)} branches open. Total workstations available: {total_wkt}."
                    else:
                        context = "Live data format unexpected. Using fallback information."
                    return {"retrieved_context": context}
                else:
                    logger.error(f"API error {resp.status}")
                    return {"retrieved_context": "Live service temporarily unavailable. Using static hours."}
    except Exception as e:
        logger.error(f"API timeout: {e}")
        return {"retrieved_context": "Network error. Please try again later."}

# ----------------------------------------------------------------------
# Generate answer (deterministic LLM with system prompt)
# ----------------------------------------------------------------------
async def generate_answer_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Generate Answer")
    request_type = state.get("request_type", "normal_info")
    question = state["messages"][-1].content

    if request_type == "normal_info":
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful library assistant. Keep answers short and friendly. Do not answer library‑specific factual questions (like hours or policies) – just chat."),
            ("human", "{input}")
        ])
        chain = prompt | llm_chat
        response = await chain.ainvoke({"input": question})
        return {"messages": [AIMessage(content=response.content)]}

    # RAG or tool answer: deterministic LLM with context
    context = state.get("retrieved_context", "")
    library_name = state.get("current_library_name")
    library_code = state.get("current_library_code")
    current_time = state.get("current_datetime") or get_current_datetime()
    user_memory = state.get("user_memory", {})

    location_hint = ""
    if library_name:
        location_hint = f"The user is currently at or near **{library_name}** (code: {library_code}). " \
                        f"If they ask about 'the library' or a branch without naming it, assume they mean this library.\n"
    date_hint = f"Current date and time: {current_time}. " \
                "If the user asks for today's hours or events, use this information.\n"
    memory_hint = ""
    if user_memory:
        memory_hint = f"User context: {json.dumps(user_memory, indent=2)}\n"

    system_prompt = f"""You are the official HKPL (Hong Kong Public Libraries) assistant.  
{date_hint}{location_hint}{memory_hint}Your answers must be based **only** on the provided context. Do not use any external or prior knowledge.

**Guidelines:**
1. If the context contains the exact answer, state it clearly and concisely.
2. If the context has **partial** information, provide what is available and politely note what is missing (e.g., "The context does not specify …").
3. If the context is completely empty or irrelevant, say:  
   *"I'm sorry, I couldn't find that information in the library's knowledge base. Please check the HKPL website or ask a librarian."*
4. Never invent facts, hours, floor numbers, policies, or any details not explicitly stated in the context.
5. Keep answers short (1-3 sentences). Avoid repeating long chunks verbatim.

Context: {context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    chain = prompt | llm_deterministic
    response = await chain.ainvoke({"question": question})
    return {"messages": [AIMessage(content=response.content)]}

# ----------------------------------------------------------------------
# Output safety filter & MCP tool node
# ----------------------------------------------------------------------
async def output_safety_filter_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Output Safety Filter")
    answer = state["messages"][-1].content
    blocked_phrases = ["unapproved", "harmful", "illegal", "self-harm", "suicide", "kill yourself"]
    if any(phrase in answer.lower() for phrase in blocked_phrases):
        return {"is_output_safe": False, "messages": [AIMessage(content="I cannot provide that answer. Please contact library staff or call the Samaritans at 2896 0000 for immediate help.")]}
    return {"is_output_safe": True}

async def mcp_tool_node(state: LibraryBotState) -> dict:
    logger.info(f"[Node] MCP Tool Node")
    tool_name = state.get("tool_name")
    tool_args = state.get("tool_args", {})
    if not tool_name:
        logger.error("mcp_tool_node called without tool_name")
        return {"retrieved_context": "No tool specified."}
    logger.info(f"[Node] MCP Tool Call: {tool_name} with args {tool_args}")
    try:
        async with streamablehttp_client(MCP_SERVER_URL) as transport:
            async with ClientSession(transport[0], transport[1]) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=tool_args)
                answer = result.content[0].text if result.content else "No output"
                return {"retrieved_context": answer}
    except Exception as e:
        logger.error(f"MCP call failed: {e}")
        return {"retrieved_context": f"Sorry, the tool '{tool_name}' is currently unavailable."}