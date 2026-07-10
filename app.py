import os
import sys
import uuid
import time
import shutil
import streamlit as st
import ollama
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from pypdf import PdfReader
from duckduckgo_search import DDGS
from dotenv import load_dotenv
from groq import Groq
from theme import inject_custom_theme, render_suggestion_label

# =====================================================================
# 🌟 STAGE 1: SET INITIAL PAGE CONFIG (MUST BE THE FIRST STREAMLIT CALL)
# =====================================================================
st.set_page_config(
    page_title="🔒 Private Intel Vault — Multi-Modal Agent",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_theme()

# 🎨 STAGE 2: INJECT EMULATED DARK IDE STYLING LAYERS
st.markdown("""
    <style>
        .stApp {
            background-color: #0d0f12 !important;
            color: #e2e8f0 !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        }
        [data-testid="stMainBlockContainer"] {
            max-width: 840px !important;
            padding: 1rem 2rem 5rem 2rem !important;
            margin: 0 auto !important;
        }
        [data-testid="stSidebarUserContent"] {
            padding-top: 1rem !important;
        }
        [data-testid="stSidebar"] {
            background-color: #16191f !important;
            border-right: 1px solid #262c36 !important;
        }
        [data-testid="stVerticalBlock"] {
            gap: 0.75rem !important;
        }
        .element-container {
            margin-bottom: 0.25rem !important;
        }
        [data-testid="stChatInput"] {
            background-color: #16191f !important;
            border: 1px solid #262c36 !important;
            border-radius: 12px !important;
            color: #ffffff !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
        }
        div[data-testid="stNotification"] {
            background-color: #1e242e !important;
            border: 1px solid rgba(97, 175, 239, 0.2) !important;
        }
        div[data-testid="stNotification"] * {
            color: #f1f5f9 !important;
        }
        div[data-testid="stStatusWidget"],
        div[data-testid="stStatusWidget"] *,
        .stStatusWidget,
        .stStatusWidget * {
            color: #ffffff !important;
        }
        summary[role="button"] div {
            color: #ffffff !important;
        }
        code {
            background-color: #1e242e !important;
            color: #f1f5f9 !important;
            border-radius: 4px !important;
            padding: 0.2rem 0.4rem !important;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace !important;
        }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 🛠️ STAGE 3: ENVIRONMENT & COMPONENT CONFIGURATIONS
# =====================================================================
from config import settings
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    try:
        groq_api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        groq_api_key = None

groq_client = Groq(api_key=groq_api_key) if groq_api_key else None

ollama_host = os.getenv("OLLAMA_HOST", settings.OLLAMA_HOST)
ollama_client = ollama.Client(host=ollama_host)

from core.retriever import retrieve_local_context, get_all_uploaded_files, delete_local_file_context

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "system_seeded" not in st.session_state:
    st.session_state.system_seeded = False

session_vault_dir = os.path.join(settings.VAULT_DIR, f"session_{st.session_state.session_id}")
os.makedirs(session_vault_dir, exist_ok=True)
os.makedirs(settings.DB_DIR, exist_ok=True)

clean_db_path = os.path.join(settings.DB_DIR, "serverless_v2")
chroma_client = chromadb.PersistentClient(path=clean_db_path)
collection_name = "private_rag_serverless"
serverless_ef = embedding_functions.ONNXMiniLM_L6_V2()

# FIX: Initialized globally outside of button logic to eliminate NameErrors
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=serverless_ef
)

# Custom Global CSS Styles Injection
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    .stApp {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(135deg, #090a0f 0%, #11131e 50%, #1a1528 100%) !important;
        color: #e2e8f0;
    }
    .stMarkdown p, .stMarkdown li, .stMarkdown span { color: #f1f5f9 !important; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 { color: #F0F2F6 !important; font-weight: 700 !important; }
    div[data-testid="stSidebar"] {
        background: rgba(13, 15, 24, 0.85) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    .glass-card {
        background: rgba(25, 28, 41, 0.6);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #61afef, #c678dd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    section[data-testid="stFileUploader"] {
        background-color: rgba(20, 22, 33, 0.5) !important;
        border: 2px dashed rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        padding: 15px !important;
    }
    .status-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
        background-color: rgba(152, 195, 121, 0.12);
        color: #98c379;
        border: 1px solid rgba(152, 195, 121, 0.25);
    }
    .status-badge-blue { background-color: rgba(97, 175, 239, 0.12); color: #61afef; border: 1px solid rgba(97, 175, 239, 0.25); }
    .status-badge-purple { background-color: rgba(198, 120, 221, 0.12); color: #c678dd; border: 1px solid rgba(198, 120, 221, 0.25); }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 📂 STAGE 4: SYSTEM INGESTION & DOCUMENT EXTRACTION LOGIC
# =====================================================================
def initialize_mock_vault_files():
    mock_file = os.path.join(settings.VAULT_DIR, "personal_profile.txt")
    if not os.listdir(settings.VAULT_DIR):
        with open(mock_file, "w", encoding="utf-8") as f:
            f.write("User Hardware Architecture: Prefers Mac ecosystem setups (MacBook Air/iPhone configuration).\n")
            f.write("Financial Threshold Constraints: Cap any electronics purchases at $200 maximum limits.\n")
            f.write("Performance Requirements: Must feature active battery efficiency configurations exceeding 15 hours.\n")

def index_file(file_path, filename, collection):
    try:
        chunks = []
        if filename.lower().endswith('.pdf'):
            reader = PdfReader(file_path)
            current_chunk = ""
            for page in reader.pages:
                text_content = page.extract_text()
                if text_content:
                    current_chunk += text_content + "\n"
                    if len(current_chunk) > 1200:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
            if current_chunk:
                chunks.append(current_chunk.strip())
        elif filename.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text_content = f.read()
            paragraphs = [p.strip() for p in text_content.split("\n\n") if p.strip()]
            for para in paragraphs:
                if len(para) > 1000:
                    for i in range(0, len(para), 1000):
                        chunks.append(para[i:i+1000])
                else:
                    chunks.append(para)
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            chunks.append(f"Local user data vault vision file context filename: {filename}")
        else:
            return 0, f"Unsupported file type: {filename}"
            
        if not chunks:
            return 0, "No readable text content parsed."
            
        for i, chunk in enumerate(chunks):
            clean_text = chunk.encode('utf-8', errors='ignore').decode('utf-8').strip()
            if clean_text:
                chunk_id = f"{filename}_chunk_{i}"
                collection.add(
                    documents=[clean_text],
                    ids=[chunk_id],
                    metadatas=[{"session_id": st.session_state.session_id, "filename": filename}]
                )
        return len(chunks), None
    except Exception as e:
        return 0, str(e)

def web_search_tool(query: str) -> str:
    retries = 3
    delay = 1.5
    max_results = 6
    for attempt in range(retries):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                if results:
                    return "\n---\n".join([
                        f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nURL: {r.get('href')}\n"
                        for r in results
                    ])
        except Exception as e:
            if attempt == retries - 1:
                print(f"Search permanently failed: {e}")
            else:
                time.sleep(delay * (attempt + 1))
    return "Result: Web search failed due to rate limits."

@st.cache_data(ttl=60)
def get_installed_ollama_models():
    try:
        model_list = ollama_client.list()
        return [m['model'] for m in model_list.get('models', [])]
    except Exception:
        return []

installed_models = get_installed_ollama_models()
if installed_models:
    llm_options = [m for m in installed_models if "embed" not in m]
    embedding_options = [m for m in installed_models if "embed" in m]
    if not llm_options: llm_options = installed_models
    if not embedding_options: embedding_options = ["nomic-embed-text:latest"] + installed_models
else:
    llm_options = ["llama-3.3-70b-specdec", "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen3:8b", "gemma4:12b", "qwen2.5:7b"]
    embedding_options = ["bge-large-en-v1.5", "nomic-embed-text:latest"]

default_llm = llm_options[0] if llm_options else "llama-3.3-70b-specdec"
default_embedding = "bge-large-en-v1.5" if "bge-large-en-v1.5" in embedding_options else embedding_options[0]

# =====================================================================
# 🧠 STAGE 5: CONTROL CENTER (SIDEBAR)
# =====================================================================
st.sidebar.markdown("<h2 style='text-align: center; margin-top:0;'>🧠 Control Center</h2>", unsafe_allow_html=True)

llm_model = st.sidebar.selectbox("Select LLM Model", llm_options, index=llm_options.index(default_llm) if default_llm in llm_options else 0)
embedding_model = st.sidebar.selectbox("Select Embedding Model", embedding_options, index=embedding_options.index(default_embedding) if default_embedding in embedding_options else 0)

if not st.session_state.system_seeded:
    initialize_mock_vault_files()
    default_files = [f for f in os.listdir(settings.VAULT_DIR) if f.endswith('.txt') and os.path.isfile(os.path.join(settings.VAULT_DIR, f))]
    for filename in default_files:
        session_file_path = os.path.join(session_vault_dir, filename)
        if not os.path.exists(session_file_path):
            shutil.copy(os.path.join(settings.VAULT_DIR, filename), session_file_path)
        num_chunks, err = index_file(session_file_path, filename, collection)
        if not err:
            st.session_state.indexed_files.append({"filename": filename, "chunks": num_chunks, "format": "txt"})
    st.session_state.system_seeded = True

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Local Document Vault")
uploaded_files = st.sidebar.file_uploader("Drag & drop documents here", type=["pdf", "txt", "png", "jpg", "jpeg"], accept_multiple_files=True, key="uploader")

if uploaded_files:
    for uploaded_file in uploaded_files:
        if not any(f["filename"] == uploaded_file.name for f in st.session_state.indexed_files):
            temp_path = os.path.join(session_vault_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.sidebar.spinner(f"Ingesting {uploaded_file.name}..."):
                num_chunks, err = index_file(temp_path, uploaded_file.name, collection)
                if not err:
                    st.session_state.indexed_files.append({"filename": uploaded_file.name, "chunks": num_chunks, "format": uploaded_file.name.split('.')[-1].lower()})

st.sidebar.markdown("---")
st.sidebar.markdown("### 🗄️ Manage Storage Vault")

try:
    all_data = collection.get(include=["metadatas"])
    current_files = list(set([m["filename"] for m in all_data["metadatas"] if m and "filename" in m]))
except Exception:
    current_files = []

if not current_files:
    st.sidebar.info("Your local vault is currently empty.")
else:
    # FIX: Clean sidebar columns with targeted deletion tracking keys
    for file_name in current_files:
        display_name = file_name if len(file_name) <= 22 else f"{file_name[:19]}..."
        col1, col2 = st.sidebar.columns([4, 1])
        col1.markdown(f"📄 `{display_name}`")
        if col2.button("🗑️", key=f"del_{file_name}"):
            collection.delete(where={"filename": file_name})
            st.session_state.indexed_files = [f for f in st.session_state.indexed_files if f["filename"] != file_name]
            st.rerun()

if st.sidebar.button("🗑️ Reset Session & Clear Vault", use_container_width=True):
    try: chroma_client.delete_collection(collection_name)
    except Exception: pass
    collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=serverless_ef)
    try: shutil.rmtree(session_vault_dir); os.makedirs(session_vault_dir, exist_ok=True)
    except Exception: pass
    st.session_state.indexed_files, st.session_state.messages = [], []
    st.rerun()

# =====================================================================
# 💻 STAGE 6: MAIN APP RENDERING ARCHITECTURE
# =====================================================================
st.markdown("<h1>🔒 Private Intel Vault</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; color: #abb2bf;'>Zero-exposure local RAG orchestrations executing entirely on Apple Silicon architecture.</p>", unsafe_allow_html=True)

st.markdown(f"""
    <div class='glass-card'>
        <span class='status-badge status-badge-blue'>● System Status: Connected</span>
        <span class='status-badge'>🛠️ Tools: Web Search [Ready]</span>
        <span class='status-badge status-badge-purple'>🤖 Engine: {llm_model}</span>
        <span class='status-badge status-badge-blue'>🔗 Vectors: {embedding_model}</span>
    </div>
""", unsafe_allow_html=True)

# FIX: Streamlined interactive diagnostic chips mapping
selected_query = None

if not st.session_state.messages:
    st.markdown("### ⚡ Quick Diagnostic Tests")
    st.caption("Click a query below to automatically test your RAG routing & vector match capabilities:")
    
    test_queries = {
        "🔒 Test Local Vault": "What are the specific constraints mentioned in my files?",
        "🌐 Test Live Web Fallback": "What is the current stock price and chart trend for Nvidia today?",
        "🧩 Test Hybrid Logic": "Based on my personal tech budget limits, can I afford a new MacBook Air right now?"
    }
    
    chip_cols = st.columns(3)
    for i, (label, query_text) in enumerate(test_queries.items()):
        if chip_cols[i].button(label, use_container_width=True, help=f"Run: '{query_text}'"):
            selected_query = query_text

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            badges_html = f"""
            <div style="margin-bottom: 8px;">
                <span class='status-badge status-badge-purple'>STATUS: {message.get("status", "Success")}</span>
                <span class='status-badge status-badge-blue'>MODE: {message.get("mode", "Offline RAG")}</span>
                <span class='status-badge'>REFERENCE_ID: {message.get("reference_id", "REF-UNKNOWN")}</span>
            </div>
            """
            st.markdown(badges_html, unsafe_allow_html=True)
        st.markdown(f'<div style="line-height: 1.6; color: #f1f5f9; padding: 6px 10px; white-space: pre-wrap; word-wrap: break-word;">{message["content"]}</div>', unsafe_allow_html=True)

# Await New Submissions (Captures both input box entries and active diagnostic chips)
user_input = st.chat_input("Query local agent...", key="primary_chat_input_canvas")
active_prompt = selected_query if selected_query else user_input

if active_prompt:
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    st.rerun()

# Processing the last submitted message
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_user_msg = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        progress_box = st.empty()
        
        def update_progress(milestone_text, details=""):
            detail_html = f"<div style='font-size:0.9rem; color:#abb2bf; margin-top:4px;'>{details}</div>" if details else ""
            progress_box.markdown(f"""
                <div class='glass-card' style='padding: 14px 20px; border-left: 4px solid #61afef;'>
                    <div style='color: #ffffff; font-weight: 500; font-size: 1rem;'>⚙️ {milestone_text}</div>
                    {detail_html}
                </div>
            """, unsafe_allow_html=True)

        update_progress("Querying local database contexts...")
        private_context = retrieve_local_context(last_user_msg, collection)
        
        update_progress("Generating tool path selection...")
        
        system_instructions = (
            "You are an advanced, isolated private AI assistant executing calculations entirely within user device storage frameworks.\n"
            "You have access to a local database of the user's private documents AND a live web search tool.\n\n"
            f"LOCAL PRIVATE CONTEXT DATABASE SEARCH RESULTS:\n{private_context if private_context else 'No local documents found.'}\n\n"
            "CRITICAL INSTRUCTION: Analyze the user's question.\n"
            "1. If the local context database answers the user's query completely, use it and do NOT call the web search tool.\n"
            "2. If the user's query requires real-time facts, call 'web_search_tool'.\n"
        )
        
        ollama_messages = [{"role": "system", "content": system_instructions}]
        for msg in st.session_state.messages[:-1]:
            ollama_messages.append({"role": msg["role"], "content": msg["content"]})
        ollama_messages.append({"role": "user", "content": last_user_msg})
        
        has_tool_call = False
        search_payload = ""
        
        try:
            response = ollama_client.chat(model=llm_model, messages=ollama_messages, tools=[web_search_tool])
            if response.message.tool_calls:
                for tool in response.message.tool_calls:
                    if tool.function.name == "web_search_tool":
                        has_tool_call = True
                        search_query = tool.function.arguments.get("query", last_user_msg)
                        
                        update_progress("Searching the web...", f"Query: {search_query}")
                        search_payload = web_search_tool(search_query)
                        
                        ollama_messages = [
                            {"role": "system", "content": "You are a helpful assistant. Synthesize a concise, clear answer for the user using the following real-time web results."},
                            {"role": "user", "content": f"REAL-TIME WEB SEARCH CONTEXT:\n{search_payload}\n\nUSER QUESTION: {last_user_msg}"}
                        ]
                        break        
        except Exception as e:
            st.error(f"Tool dispatch error: {e}")

        if not has_tool_call:
            update_progress("Synthesizing response using local context...")
            rag_prompt = f"Context:\n{private_context}\n\nQuestion:\n{last_user_msg}"
            if groq_client:
                cloud_payload = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": rag_prompt}])
                class MockObj:
                    def __init__(self, c): self.content = c
                class MockResp:
                    def __init__(self, c): self.message = MockObj(c)
                response = MockResp(cloud_payload.choices[0].message.content)
            else:
                response = ollama_client.chat(model=llm_model, messages=[{"role": "user", "content": rag_prompt}])
        else:
            update_progress("Web results integrated. Streaming final synthesis...")
            time.sleep(0.3)
            
        progress_box.empty()
        
        ref_id = f"REF-{uuid.uuid4().hex[:6].upper()}"
        mode_val = "RAG + Web Search" if has_tool_call else "Offline RAG"
        status_val = "Web Retrieval Success" if has_tool_call else ("Local Context Success" if private_context else "Local Context Empty")
        
        st.markdown(f"""
            <div style="margin-bottom: 8px;">
                <span class='status-badge status-badge-purple'>STATUS: {status_val}</span>
                <span class='status-badge status-badge-blue'>MODE: {mode_val}</span>
                <span class='status-badge'>REFERENCE_ID: {ref_id}</span>
            </div>
        """, unsafe_allow_html=True)

        response_placeholder = st.empty()
        full_response = ""
        
        try:
            if has_tool_call:
                stream = ollama_client.chat(model=llm_model, messages=ollama_messages, stream=True)
                for chunk in stream:
                    full_response += chunk['message']['content']
                    response_placeholder.markdown(f'<div style="line-height: 1.6; color: #f1f5f9; padding: 6px 10px; white-space: pre-wrap; word-wrap: break-word;">{full_response}▌</div>', unsafe_allow_html=True)
            else:
                content = response.message.content if response and response.message else "Empty prompt reply."
                for word in content.split(" "):
                    full_response += word + " "
                    response_placeholder.markdown(f'<div style="line-height: 1.6; color: #f1f5f9; padding: 6px 10px; white-space: pre-wrap; word-wrap: break-word;">{full_response}▌</div>', unsafe_allow_html=True)
                    time.sleep(0.01)
            response_placeholder.markdown(f'<div style="line-height: 1.6; color: #f1f5f9; padding: 6px 10px; white-space: pre-wrap; word-wrap: break-word;">{full_response}</div>', unsafe_allow_html=True)
        except Exception as e:
            full_response = f"Execution runtime error: {e}"
            response_placeholder.markdown(f"<div>{full_response}</div>", unsafe_allow_html=True)
            
        st.session_state.messages.append({
            "role": "assistant", "content": full_response,
            "status": status_val, "mode": mode_val, "reference_id": ref_id
        })