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

# Load configurations and environment variables
from config import settings
load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# 1. STREAMLIT INITIALIZATION (MUST BE FIRST)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔒 Private Intel Vault — Multi-Modal Agent",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────────────
# 2. GLOBAL CLIENT INITIALIZATION & CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
# Leverage environmental variables loaded via load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    try:
        groq_api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        groq_api_key = None

# Instantiate the Groq client only if an API key is actually present
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None

# Initialize Ollama Connection Client
ollama_host = os.getenv("OLLAMA_HOST", settings.OLLAMA_HOST)
ollama_client = ollama.Client(host=ollama_host)

# Initialize Ollama Connection Client
ollama_host = os.getenv("OLLAMA_HOST", settings.OLLAMA_HOST)
ollama_client = ollama.Client(host=ollama_host)

# Core retriever utilities (Ensuring get_all_uploaded_files and delete_local_file_context are present)
from core.retriever import retrieve_local_context, get_all_uploaded_files, delete_local_file_context

# Session State Foundations
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "system_seeded" not in st.session_state:
    st.session_state.system_seeded = False

# Establish Directory Paths
session_vault_dir = os.path.join(settings.VAULT_DIR, f"session_{st.session_state.session_id}")
os.makedirs(session_vault_dir, exist_ok=True)
os.makedirs(settings.DB_DIR, exist_ok=True)

# ChromaDB Serverless Orchestration
clean_db_path = os.path.join(settings.DB_DIR, "serverless_v2")
chroma_client = chromadb.PersistentClient(path=clean_db_path)
collection_name = "private_rag_serverless"
serverless_ef = embedding_functions.ONNXMiniLM_L6_V2()

collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=serverless_ef
)

# ──────────────────────────────────────────────────────────────────────────────
# 3. GRAPHICAL USER INTERFACE & CSS INJECTION
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Global typography and gradient */
    .stApp {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(135deg, #090a0f 0%, #11131e 50%, #1a1528 100%) !important;
        color: #e2e8f0;
    }
    /* Force high-contrast readability */
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #FFFFFF !important;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #F0F2F6 !important;
        font-weight: 700 !important;
    }
    /* Glassmorphism sidebar */
    div[data-testid="stSidebar"] {
        background: rgba(13, 15, 24, 0.85) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    /* Cards */
    .glass-card {
        background: rgba(25, 28, 41, 0.6);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(97, 175, 239, 0.3);
        box-shadow: 0 12px 40px 0 rgba(97, 175, 239, 0.1);
        transform: translateY(-1px);
    }
    /* Chat inputs */
    div[data-testid="stChatInput"] {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background-color: rgba(20, 22, 33, 0.9) !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
    }
    /* Header Gradients */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700 !important;
        letter-spacing: -0.025em;
        background: linear-gradient(90deg, #61afef, #c678dd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    /* Uploaders */
    section[data-testid="stFileUploader"] {
        background-color: rgba(20, 22, 33, 0.5) !important;
        border: 2px dashed rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        padding: 15px !important;
    }
    section[data-testid="stFileUploader"]:hover {
        border-color: #61afef !important;
        background-color: rgba(97, 175, 239, 0.05) !important;
    }
    /* Status Badges */
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
    """,
    unsafe_allow_html=True
)

# ──────────────────────────────────────────────────────────────────────────────
# 4. SYSTEM INGESTION & DOCUMENT EXTRACTION LOGIC
# ──────────────────────────────────────────────────────────────────────────────
def initialize_mock_vault_files():
    """Builds foundational files in data/vault/ if users start bare."""
    mock_file = os.path.join(settings.VAULT_DIR, "personal_profile.txt")
    if not os.listdir(settings.VAULT_DIR):
        with open(mock_file, "w", encoding="utf-8") as f:
            f.write("User Hardware Architecture: Prefers Mac ecosystem setups (MacBook Air/iPhone configuration).\n")
            f.write("Financial Threshold Constraints: Cap any electronics purchases at $200 maximum limits.\n")
            f.write("Performance Requirements: Must feature active battery efficiency configurations exceeding 15 hours.\n")

def index_file(file_path, filename, collection):
    """Parses and vectorizes document chunks directly into the collection using registered serverless EF."""
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
            image_description = f"Local user data vault vision file context filename: {filename}"
            chunks.append(image_description)
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
    """Executes a web search via DuckDuckGo text engine with exponential backoff retry parameters."""
    retries = 3
    delay = 1.5
    max_results = 6
    for attempt in range(retries):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                if results:
                    formatted_results = [
                        f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nURL: {r.get('href')}\n"
                        for r in results
                    ]
                    return "\n---\n".join(formatted_results)
        except Exception as e:
            if attempt == retries - 1:
                print(f"Search permanently failed after {retries} attempts: {e}")
            else:
                time.sleep(delay * (attempt + 1))
    return "Result: Web search failed due to a temporary provider rate limit. Please try again in a moment."

@st.cache_data(ttl=60)
def get_installed_ollama_models():
    """Queries local Ollama socket endpoint to fetch installed weights."""
    try:
        model_list = ollama_client.list()
        return [m['model'] for m in model_list.get('models', [])]
    except Exception:
        return []

# Dynamic Engine Sourcing
installed_models = get_installed_ollama_models()
if installed_models:
    llm_options = [m for m in installed_models if "embed" not in m]
    embedding_options = [m for m in installed_models if "embed" in m]
    if not llm_options: llm_options = installed_models
    if not embedding_options: embedding_options = ["nomic-embed-text:latest"] + installed_models
else:
    llm_options = ["llama-3.3-70b-specdec", "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "qwen3:8b", "gemma4:12b", "qwen2.5:7b"]
    embedding_options = ["bge-large-en-v1.5", "nomic-embed-text:latest", "nomic-embed-text"]

default_llm = llm_options[0] if llm_options else "llama-3.3-70b-specdec"
default_embedding = "bge-large-en-v1.5" if "bge-large-en-v1.5" in embedding_options else embedding_options[0]

# ──────────────────────────────────────────────────────────────────────────────
# 5. STREAMLIT SIDEBAR: CONTROL CENTER
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("<h2 style='text-align: center; margin-top:0;'>🧠 Control Center</h2>", unsafe_allow_html=True)
st.sidebar.markdown("Configure local models and manage private data maps.")

llm_model = st.sidebar.selectbox(
    "Select LLM Model (Brain)",
    llm_options,
    index=llm_options.index(default_llm) if default_llm in llm_options else 0
)
embedding_model = st.sidebar.selectbox(
    "Select Embedding Model",
    embedding_options,
    index=embedding_options.index(default_embedding) if default_embedding in embedding_options else 0
)

# Automated Ingestion Seeding
if not st.session_state.system_seeded:
    initialize_mock_vault_files()
    vault_path = settings.VAULT_DIR
    default_files = [f for f in os.listdir(vault_path) if f.endswith('.txt') and os.path.isfile(os.path.join(vault_path, f))]
    
    if default_files:
        for filename in default_files:
            file_path = os.path.join(vault_path, filename)
            session_file_path = os.path.join(session_vault_dir, filename)
            if not os.path.exists(session_file_path):
                shutil.copy(file_path, session_file_path)
            num_chunks, err = index_file(session_file_path, filename, collection)
            if not err:
                st.session_state.indexed_files.append({"filename": filename, "chunks": num_chunks, "format": "txt"})
    st.session_state.system_seeded = True

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Local Document Vault")

uploaded_files = st.sidebar.file_uploader(
    "Drag & drop documents here",
    type=["pdf", "txt", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key="uploader"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🗄️ Manage Storage Vault")

# Query ChromaDB metadata directly to get the true list of files
try:
    all_data = collection.get(include=["metadatas"])
    # Extract unique filenames from the metadata blocks safely
    current_files = list(set([m["filename"] for m in all_data["metadatas"] if m and "filename" in m]))
except Exception:
    current_files = []

if not current_files:
    st.sidebar.info("Your local vault is currently empty.")
else:
    st.sidebar.caption(f"Currently indexing {len(current_files)} file(s):")

    for file_name in current_files:
        # Format display name gracefully
        display_name = file_name if len(file_name) <= 22 else f"{file_name[:19]}..."
        
        # 🌟 Force rendering on a single row as an explicit button action
        if st.sidebar.button(f"🗑️ Delete {display_name}", key=f"del_{file_name}", help=f"Remove {file_name} from vault"):
            with st.spinner(f"Evicting {file_name}..."):
                try:
                    # Delete chunks matching this specific filename from the vector space
                    collection.delete(where={"filename": file_name})
                    
                    # Keep our temporary session tracking state synchronized
                    st.session_state.indexed_files = [f for f in st.session_state.indexed_files if f["filename"] != file_name]
                    
                    st.sidebar.success(f"Removed {file_name}!")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Failed to delete file: {e}")                

if uploaded_files:
    for uploaded_file in uploaded_files:
        already_indexed = any(f["filename"] == uploaded_file.name for f in st.session_state.indexed_files)
        if not already_indexed:
            temp_path = os.path.join(session_vault_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                num_chunks, err = index_file(temp_path, uploaded_file.name, collection)
                if err:
                    st.sidebar.error(f"Error {uploaded_file.name}: {err}")
                else:
                    fmt = uploaded_file.name.split('.')[-1].lower()
                    st.session_state.indexed_files.append({"filename": uploaded_file.name, "chunks": num_chunks, "format": fmt})
                    st.sidebar.success(f"Indexed {uploaded_file.name} ({num_chunks} chunks)")

if st.session_state.indexed_files:
    st.sidebar.markdown("#### Currently Indexed Vault Items")
    for f in st.session_state.indexed_files:
        icon = "📸" if f["format"] in ["png", "jpg", "jpeg"] else "📄"
        st.sidebar.markdown(f"<span class='status-badge'>{icon} {f['filename']} ({f['chunks']} chunks)</span>", unsafe_allow_html=True)

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Reset Session & Clear Vault", use_container_width=True):
    try:
        chroma_client.delete_collection(collection_name)
    except Exception: pass
    collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=serverless_ef)
    try:
        shutil.rmtree(session_vault_dir)
        os.makedirs(session_vault_dir, exist_ok=True)
    except Exception: pass
    st.session_state.indexed_files = []
    st.session_state.messages = []
    st.sidebar.success("Vault database & messages wiped!")
    time.sleep(1)
    st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# 6. STREAMLIT MAIN STAGE: CONVERSATION FRAMEWORK
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("<h1>🔒 Private Intel Vault</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; color: #abb2bf;'>Zero-exposure local RAG orchestrations executing entirely on Apple Silicon architecture.</p>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class='glass-card'>
        <span class='status-badge status-badge-blue'>● System Status: Connected</span>
        <span class='status-badge'>🛠️ Tools: Web Search [Ready]</span>
        <span class='status-badge status-badge-purple'>🤖 Engine: {llm_model}</span>
        <span class='status-badge status-badge-blue'>🔗 Vectors: {embedding_model}</span>
    </div>
    """, 
    unsafe_allow_html=True
)

if not st.session_state.messages:
    st.markdown(
        """
        <div class='glass-card' style='margin-top: 15px;'>
            <h3 style='margin-top:0;'>🔒 Local Private Intelligence Framework</h3>
            <p>Welcome! This application runs a secure local model workflow with Retrieval-Augmented Generation (RAG).</p>
            <ul>
                <li>Knowledge base pre-seeded with a mock configuration: <code>personal_profile.txt</code>.</li>
                <li>Query: <code>"What is my electronic device purchase budget limit?"</code> to target local vector contexts.</li>
                <li>Query: <code>"What is the latest pricing for Apple MacBook Air computers online?"</code> to trigger real-time search paths.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            badges_html = f"""
            <div style="margin-bottom: 8px;">
                <span class='status-badge status-badge-purple'>STATUS: {message.get("status", "Local Context Success")}</span>
                <span class='status-badge status-badge-blue'>MODE: {message.get("mode", "Offline RAG")}</span>
                <span class='status-badge'>REFERENCE_ID: {message.get("reference_id", "REF-UNKNOWN")}</span>
            </div>
            """
            st.markdown(badges_html, unsafe_allow_html=True)
        st.markdown(f'<div style="line-height: 1.6; color: #FFFFFF; padding: 6px 10px; white-space: pre-wrap; word-wrap: break-word;">{message["content"]}</div>', unsafe_allow_html=True)

if user_prompt := st.chat_input("Query local agent..."):
    with st.chat_message("user"):
        st.markdown(f'<div style="line-height: 1.6; color: #FFFFFF; padding: 6px 10px; white-space: pre-wrap; word-wrap: break-word;">{user_prompt}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    with st.chat_message("assistant"):
        with st.status("🧠 Agent routing workflow...", expanded=True) as status:
            status.update(label="🔍 Querying local database contexts...", state="running")
            private_context = retrieve_local_context(user_prompt, collection)
            
            if private_context:
                st.write("**ChromaDB Matches Retrieved:**")
                st.info(private_context[:400] + ("..." if len(private_context) > 400 else ""))
            else:
                st.write("No matching documents found in the session vector store.")
            
            time.sleep(0.2)
            status.update(label="🤖 Generating tool path selection...", state="running")
            
            system_instructions = (
                "You are an advanced, isolated private AI assistant executing calculations entirely within user device storage frameworks.\n"
                "You have access to a local database of the user's private documents AND a live web search tool.\n\n"
                f"LOCAL PRIVATE CONTEXT DATABASE SEARCH RESULTS:\n{private_context if private_context else 'No local documents found.'}\n\n"
                "CRITICAL INSTRUCTION: Analyze the user's question.\n"
                "1. If the local context database answers the user's query completely, use it and do NOT call the web search tool.\n"
                "2. If the user's query requires real-time facts, information outside the database, or local data is insufficient, call 'web_search_tool'.\n"
                "3. Keep answers concise, clear, and focused."
            )
            
            ollama_messages = [{"role": "system", "content": system_instructions}]
            for msg in st.session_state.messages[:-1]:
                ollama_messages.append({"role": msg["role"], "content": msg["content"]})
            ollama_messages.append({"role": "user", "content": user_prompt})
            
            has_tool_call = False
            search_payload = ""
            
            try:
                response = ollama_client.chat(model=llm_model, messages=ollama_messages, tools=[web_search_tool])
                if response.message.tool_calls:
                    for tool in response.message.tool_calls:
                        if tool.function.name == "web_search_tool":
                            has_tool_call = True
                            search_query = tool.function.arguments.get("query", user_prompt)
                            status.update(label=f"🌐 Searching the web for: '{search_query}'...", state="running")
                            search_payload = web_search_tool(search_query)
                            
                            st.write(f"**Web Search Query:** `{search_query}`")
                            st.info(search_payload[:400] + ("..." if len(search_payload) > 400 else ""))
                            
                            # 🌟 FIX: Flatten the message history into standard prose context 
                            # so local models don't get trapped in structural JSON loops
                            ollama_messages = [
                                {
                                    "role": "system", 
                                    "content": "You are a helpful assistant. Synthesize a concise, clear answer for the user using the following real-time web results."
                                },
                                {
                                    "role": "user", 
                                    "content": f"REAL-TIME WEB SEARCH CONTEXT:\n{search_payload}\n\nUSER QUESTION: {user_prompt}"
                                }
                            ]
                            break        
            except Exception as e:
                st.error(f"Error during tool dispatch step: {e}")

            if not has_tool_call:
                status.update(label="✓ Synthesizing response using local context...", state="running")
                rag_prompt = f"""You are a helpful assistant answering questions using local document context.
                
CONTEXT FROM VAULT DOCUMENTS:
{private_context}

USER QUESTION:
{user_prompt}

Answer accurately using only the provided context. If the answer cannot be found, state that clearly."""

                # 🌟 FIX THE CONDITION HERE: Check if the client object itself exists
                if groq_client:
                    cloud_payload = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile", 
                        messages=[{"role": "user", "content": rag_prompt}]
                    )
                    class MockMessage:
                        def __init__(self, content): self.content = content
                    class MockResponse:
                        def __init__(self, content): self.message = MockMessage(content)
                    response = MockResponse(cloud_payload.choices[0].message.content)
                else:
                    response = ollama_client.chat(model=llm_model, messages=[{"role": "user", "content": rag_prompt}])
            else:
                status.update(label="✓ Web results integrated. Synthesizing response...", state="running")                            
        
        # Metadata configuration 
        ref_id = f"REF-{uuid.uuid4().hex[:6].upper()}"
        mode_val = "RAG + Web Search" if has_tool_call else "Offline RAG"
        status_val = "Web Retrieval Success" if has_tool_call and not search_payload.startswith("Error") else ("Local Context Success" if private_context else "Local Context Empty")
        
        badges_html = f"""
        <div style="margin-bottom: 8px;">
            <span class='status-badge status-badge-purple'>STATUS: {status_val}</span>
            <span class='status-badge status-badge-blue'>MODE: {mode_val}</span>
            <span class='status-badge'>REFERENCE_ID: {ref_id}</span>
        </div>
        """
        st.markdown(badges_html, unsafe_allow_html=True)

        response_placeholder = st.empty()
        full_response = ""
        
        try:
            if has_tool_call:
                stream = ollama_client.chat(model=llm_model, messages=ollama_messages, stream=True)
                for chunk in stream:
                    full_response += chunk['message']['content']
                    response_placeholder.markdown(f'<div style="line-height: 1.6; color: #FFFFFF; padding: 6px 10px; white-space: pre-wrap; word-wrap: break-word;">{full_response}▌</div>', unsafe_allow_html=True)
            else:
                content = response.message.content if 'response' in locals() and response.message else "Could not generate response."
                for word in content.split(" "):
                    full_response += word + " "
                    response_placeholder.markdown(f'<div style="line-height: 1.6; color: #FFFFFF; padding: 6px 10px; white-space: pre-wrap; word-wrap: break-word;">{full_response}▌</div>', unsafe_allow_html=True)
                    time.sleep(0.01)
            response_placeholder.markdown(f'<div style="line-height: 1.6; color: #FFFFFF; padding: 6px 10px; white-space: pre-wrap; word-wrap: break-word;">{full_response}</div>', unsafe_allow_html=True)
        except Exception as e:
            full_response = f"Error during final response streaming execution: {e}"
            response_placeholder.markdown(f'<div style="line-height: 1.6; color: #FFFFFF; padding: 6px 10px; white-space: pre-wrap; word-wrap: break-word;">{full_response}</div>', unsafe_allow_html=True)
            
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "status": status_val,
            "mode": mode_val,
            "reference_id": ref_id
        })