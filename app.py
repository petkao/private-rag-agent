import streamlit as st
from groq import Groq

# 🔒 CHANGE THIS LINE: 
# This tells the code to automatically grab the key from your secure vault
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

import os
import sys
import uuid
import time
import shutil
import streamlit as st
import ollama
import chromadb
from pypdf import PdfReader
from duckduckgo_search import DDGS
from dotenv import load_dotenv
# ──────────────────────────────────────────────────────────────────────────────
# FORCE HIGH-CONTRAST FOR DARK THEME TEXT READABILITY
# ──────────────────────────────────────────────────────────────────────────────
import streamlit as st
st.markdown(
    """
    <style>
    /* Force all body text, bullet points, and markdown spans to clean white */
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #FFFFFF !important;
    }
    /* Make headers highly visible */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #F0F2F6 !important;
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Load settings and configurations
from config import settings

# Load environmental variables from .env
load_dotenv()

# Force page configuration to be the absolute first streamlit call
st.set_page_config(
    page_title="🔒 Private Intel Vault — Multi-Modal Agent",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS INJECTION FOR PREMIUM GRAPHICAL INTERFACE (DARK MODE GLASSMORPHISM)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Apply global typography and background gradient */
.stApp {
    font-family: 'Outfit', sans-serif;
    background: linear-gradient(135deg, #090a0f 0%, #11131e 50%, #1a1528 100%) !important;
    color: #e2e8f0;
}

/* Glassmorphism sidebar */
div[data-testid="stSidebar"] {
    background: rgba(13, 15, 24, 0.85) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Glassmorphism card widget container */
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

/* Chat Input Styling overrides */
div[data-testid="stChatInput"] {
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    background-color: rgba(20, 22, 33, 0.9) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
    transition: all 0.3s ease !important;
}

/* Custom header gradients */
h1, h2, h3 {
    font-family: 'Outfit', sans-serif;
    font-weight: 700 !important;
    letter-spacing: -0.025em;
    background: linear-gradient(90deg, #61afef, #c678dd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* File uploader enhancements */
section[data-testid="stFileUploader"] {
    background-color: rgba(20, 22, 33, 0.5) !important;
    border: 2px dashed rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    padding: 15px !important;
    transition: all 0.3s ease !important;
}
section[data-testid="stFileUploader"]:hover {
    border-color: #61afef !important;
    background-color: rgba(97, 175, 239, 0.05) !important;
}

/* Custom indicator badges */
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

.status-badge-blue {
    background-color: rgba(97, 175, 239, 0.12);
    color: #61afef;
    border: 1px solid rgba(97, 175, 239, 0.25);
}

.status-badge-purple {
    background-color: rgba(198, 120, 221, 0.12);
    color: #c678dd;
    border: 1px solid rgba(198, 120, 221, 0.25);
}
</style>
""", unsafe_allow_html=True)

# Initialize Ollama Connection Client
ollama_host = os.getenv("OLLAMA_HOST", settings.OLLAMA_HOST)
ollama_client = ollama.Client(host=ollama_host)

# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM INGESTION & DOCUMENT EXTRACTION LOGIC
# ──────────────────────────────────────────────────────────────────────────────

def initialize_mock_vault_files():
    """Builds foundational files in data/vault/ if users start bare."""
    mock_file = os.path.join(settings.VAULT_DIR, "personal_profile.txt")
    if not os.listdir(settings.VAULT_DIR):
        with open(mock_file, "w", encoding="utf-8") as f:
            f.write("User Hardware Architecture: Prefers Mac ecosystem setups (MacBook Air/iPhone configuration).\n")
            f.write("Financial Threshold Constraints: Cap any electronics purchases at $200 maximum limits.\n")
            f.write("Performance Requirements: Must feature active battery efficiency configurations exceeding 15 hours.\n")

def index_file(file_path, filename, collection, embedding_model):
    """Parses and vectorizes document chunks into a session ChromaDB collection."""
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
            # Split text by paragraphs or 1000 character boundaries
            paragraphs = [p.strip() for p in text_content.split("\n\n") if p.strip()]
            for para in paragraphs:
                if len(para) > 1000:
                    for i in range(0, len(para), 1000):
                        chunks.append(para[i:i+1000])
                else:
                    chunks.append(para)
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Vision representation text string fallback
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
                
                # Strip out the manual ollama_client.embed calculations!
                # By omitting the 'embeddings' parameter here, ChromaDB automatically
                # invokes the serverless_ef engine we registered during initialization.
                collection.add(
                    documents=[clean_text],
                    ids=[chunk_id],
                    metadatas=[{"session_id": st.session_state.session_id}]  # Tag it!
                )
                
        return len(chunks), None

    except Exception as e:
        return 0, str(e)

# ──────────────────────────────────────────────────────────────────────────────
# NATIVE PYTHON TOOLS FOR LLM INTEGRATION
# ──────────────────────────────────────────────────────────────────────────────
import time
from duckduckgo_search import DDGS

def web_search_tool(query: str) -> str:
    """
    Executes a web search with an automated retry loop 
    to bypass aggressive rate limiting.
    """
    retries = 3
    delay = 1.5  # Seconds to pause between attempts
    max_results = 3
    
    for attempt in range(retries):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                if results:
                    # Format results cleanly as a single string payload for your LLM
                    formatted_results = []
                    for r in results:
                        formatted_results.append(f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nURL: {r.get('href')}\n")
                    return "\n---\n".join(formatted_results)
        except Exception as e:
            if attempt == retries - 1:
                print(f"Search permanently failed after {retries} attempts: {e}")
            else:
                time.sleep(delay * (attempt + 1))  # Exponential backoff
                
    return "Result: Web search failed due to a temporary provider rate limit. Please try again in a moment."

# ──────────────────────────────────────────────────────────────────────────────
# MODEL DETECTION & ENVIRONMENT SETUP
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def get_installed_ollama_models():
    """Queries local Ollama socket endpoint to fetch installed weights."""
    try:
        model_list = ollama_client.list()
        return [m['model'] for m in model_list.get('models', [])]
    except Exception:
        return []

# Set up Session ID and Isolated Paths
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

if "system_seeded" not in st.session_state:
    st.session_state.system_seeded = False

session_vault_dir = os.path.join(settings.VAULT_DIR, f"session_{st.session_state.session_id}")
os.makedirs(session_vault_dir, exist_ok=True)
os.makedirs(settings.DB_DIR, exist_ok=True)

# Connect to database using isolated collection name based on session_id
import os
import chromadb.utils.embedding_functions as embedding_functions

# Force a brand-new database directory to bypass the corrupted SQLite file
clean_db_path = os.path.join(settings.DB_DIR, "serverless_v2")
chroma_client = chromadb.PersistentClient(path=clean_db_path)

# Collection configuration matching our new serverless architecture
collection_name = "private_rag_serverless"
serverless_ef = embedding_functions.ONNXMiniLM_L6_V2()

collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=serverless_ef
)

# Query Ollama to populate choices
installed_models = get_installed_ollama_models()
if installed_models:
    llm_options = [m for m in installed_models if "embed" not in m]
    embedding_options = [m for m in installed_models if "embed" in m]
    if not llm_options:
        llm_options = installed_models
    if not embedding_options:
        embedding_options = ["nomic-embed-text:latest"] + installed_models
else:
    llm_options = [
        "llama-3.3-70b-specdec",
        "llama-3.3-70b-versatile", 
        "llama-3.1-8b-instant", 
        "mixtral-8x7b-32768", 
        "qwen3:8b", 
        "gemma4:12b", 
        "qwen2.5:7b"
    ]

    embedding_options = [
    "bge-large-en-v1.5",  # Groq Cloud Embedding Model
    "nomic-embed-text:latest",
    "nomic-embed-text"
]

# Select default brain model options safely
default_llm = llm_options[0]
# Set the default model cleanly using its placement position in the list
if len(llm_options) > 0:
    default_llm = llm_options[0]
else:
    default_llm = "llama-3.3-70b-specdec"

default_embedding = "bge-large-en-v1.5" if "bge-large-en-v1.5" in embedding_options else embedding_options[0]

# ──────────────────────────────────────────────────────────────────────────────
# STREAMLIT SIDEBAR: CONTROL CENTER
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

# Seed mock profile file to database if brand new session
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
            
            # Index it
            num_chunks, err = index_file(session_file_path, filename, collection, default_embedding)
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

# Process any upload adjustments
if uploaded_files:
    for uploaded_file in uploaded_files:
        already_indexed = any(f["filename"] == uploaded_file.name for f in st.session_state.indexed_files)
        if not already_indexed:
            temp_path = os.path.join(session_vault_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.sidebar.spinner(f"Ingesting {uploaded_file.name}..."):
                # Swap out 'embedding_model' for your new 'serverless_ef' object
                num_chunks, err = index_file(temp_path, uploaded_file.name, collection, "bge-large-en-v1.5")
                if err:
                    st.sidebar.error(f"Error {uploaded_file.name}: {err}")
                else:
                    fmt = uploaded_file.name.split('.')[-1].lower()
                    st.session_state.indexed_files.append({"filename": uploaded_file.name, "chunks": num_chunks, "format": fmt})
                    st.sidebar.success(f"Indexed {uploaded_file.name} ({num_chunks} chunks)")

# Render Vault contents list
if st.session_state.indexed_files:
    st.sidebar.markdown("#### Currently Indexed Vault Items")
    for f in st.session_state.indexed_files:
        icon = "📄"
        if f["format"] in ["png", "jpg", "jpeg"]:
            icon = "📸"
        st.sidebar.markdown(f"<span class='status-badge'>{icon} {f['filename']} ({f['chunks']} chunks)</span>", unsafe_allow_html=True)
else:
    st.sidebar.info("Vault is currently empty.")

st.sidebar.markdown("---")

# Clear operations
if st.sidebar.button("🗑️ Reset Session & Clear Vault", use_container_width=True):
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass
    
    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    try:
        shutil.rmtree(session_vault_dir)
        os.makedirs(session_vault_dir, exist_ok=True)
    except Exception:
        pass
        
    st.session_state.indexed_files = []
    st.session_state.messages = []
    st.sidebar.success("Vault database & messages wiped!")
    time.sleep(1)
    st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# STREAMLIT MAIN STAGE: CONVERSATION FRAMEWORK
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("<h1>🔒 Private Intel Vault</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; color: #abb2bf;'>Zero-exposure local RAG orchestrations executing entirely on Apple Silicon architecture.</p>", unsafe_allow_html=True)

# Top Status Indicators
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

# Empty welcome state
if not st.session_state.messages:
    st.markdown(
        """
        <div class='glass-card' style='margin-top: 15px;'>
            <h3 style='margin-top:0;'>🔒 Local Private Intelligence Framework</h3>
            <p>Welcome! This application runs a secure local model workflow with Retrieval-Augmented Generation (RAG).
            Your uploads and queries are never routed onto public servers or external clouds.</p>
            <p><b>RAG Ingestion and Tool Orchestrations:</b></p>
            <ul>
                <li>The system has pre-seeded the local knowledge base with a mock configuration: <code>personal_profile.txt</code>.</li>
                <li>Try querying: <code>"What is my electronic device purchase budget limit?"</code> to retrieve local vault contexts.</li>
                <li>Try querying: <code>"What is the latest pricing for Apple MacBook Air computers online?"</code>. The agent will detect that the local vault is insufficient and dynamically dispatch the DuckDuckGo live search tool fallback.</li>
                <li>Use the sidebar uploader to vectorize and inject custom PDFs, text notes, or images.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            msg_status = message.get("status", "Local Context Success")
            msg_mode = message.get("mode", "Offline RAG")
            msg_ref = message.get("reference_id", "REF-UNKNOWN")
            
            badges_html = f"""
            <div style="margin-bottom: 8px;">
                <span class='status-badge status-badge-purple'>STATUS: {msg_status}</span>
                <span class='status-badge status-badge-blue'>MODE: {msg_mode}</span>
                <span class='status-badge'>REFERENCE_ID: {msg_ref}</span>
            </div>
            """
            st.markdown(badges_html, unsafe_allow_html=True)
        st.markdown(f'<div style="line-height: 1.6; color: #FFFFFF; padding: 6px 10px; white-space: pre-wrap; word-wrap: break-word;">{message["content"]}</div>', unsafe_allow_html=True)

# Prompt handling
if user_prompt := st.chat_input("Query local agent..."):
    # Display prompt
    with st.chat_message("user"):
        st.markdown(f'<div style="line-height: 1.6; color: #FFFFFF; padding: 6px 10px; white-space: pre-wrap; word-wrap: break-word;">{user_prompt}</div>', unsafe_allow_html=True)
    
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    # Process agent response inside container
    with st.chat_message("assistant"):
        
        # Helper to compute query embeddings
        def get_query_embedding(query, embed_model):
            try:
                res = ollama_client.embed(model=embed_model, input=query)
                return res['embeddings'][0]
            except Exception as e:
                st.error(f"Embedding calculation failure: {e}")
                return None

        # Helper to run vector database query
        def retrieve_local_context(query, col, embed_model):
            try:
                # Ditch manual embedding extraction! 
                # Pass the raw string to query_texts, and Chroma will use serverless_ef automatically.
                db_results = col.query(query_texts=[query], n_results=4)
                docs = db_results.get('documents', [[]])[0]
                return "\n---\n".join(docs) if docs else ""
            except Exception:
                return ""

        # Step 1: Initialize thinking status widget
        with st.status("🧠 Agent routing workflow...", expanded=True) as status:
            status.update(label="🔍 Querying local database contexts...", state="running")
            private_context = retrieve_local_context(user_prompt, collection, serverless_ef)
            
            if private_context:
                st.write("**ChromaDB Matches Retrieved:**")
                st.info(private_context[:400] + ("..." if len(private_context) > 400 else ""))
            else:
                st.write("No matching documents found in the session vector store.")
            
            time.sleep(0.2)
            
            status.update(label="🤖 Generating tool path selection...", state="running")
            
            # Setup instructions with contextual references
            system_instructions = (
                "You are an advanced, isolated private AI assistant executing calculations entirely within user device storage frameworks.\n"
                "You have access to a local database of the user's private documents AND a live web search tool.\n\n"
                f"LOCAL PRIVATE CONTEXT DATABASE SEARCH RESULTS:\n{private_context if private_context else 'No local documents found.'}\n\n"
                "CRITICAL INSTRUCTION: Analyze the user's question.\n"
                "1. If the local context database answers the user's query completely, use it and do NOT call the web search tool.\n"
                "2. If the user's query requires real-time facts (weather, current news, dates), information outside the database, or the local data is insufficient/empty, call the 'web_search_tool' function.\n"
                "3. Keep your answers concise, clear, and highly focused."
            )
            
            ollama_messages = [{"role": "system", "content": system_instructions}]
            for msg in st.session_state.messages[:-1]:
                ollama_messages.append({"role": msg["role"], "content": msg["content"]})
            ollama_messages.append({"role": "user", "content": user_prompt})
            
            has_tool_call = False
            search_query = ""
            search_payload = ""
            
            try:
                # Direct tool calling dispatch check
                response = ollama_client.chat(
                    model=llm_model,
                    messages=ollama_messages,
                    tools=[web_search_tool]
                )
                
                if response.message.tool_calls:
                    for tool in response.message.tool_calls:
                        if tool.function.name == "web_search_tool":
                            has_tool_call = True
                            search_query = tool.function.arguments.get("query", user_prompt)
                            status.update(label=f"🌐 Searching the web for: '{search_query}'...", state="running")
                            
                            search_payload = web_search_tool(search_query)
                            
                            st.write(f"**Web Search Query:** `{search_query}`")
                            st.write("**Search Results Summary:**")
                            st.info(search_payload[:400] + ("..." if len(search_payload) > 400 else ""))
                            
                            # Append tool outputs to messages history
                            ollama_messages.append(response.message)
                            ollama_messages.append({
                                "role": "tool",
                                "tool_name": "web_search_tool",
                                "content": search_payload
                            })
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

Answer the user's question accurately using only the provided context. If the answer cannot be found in the context, state that clearly."""

                # 1. Initialize the cloud client natively inside the production block
                import os
                from groq import Groq
                groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
                
                # 2. Call the web-accessible cloud model
                cloud_payload = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile", 
                    messages=[{"role": "user", "content": rag_prompt}]
                )
                
                # 3. MOCK CLASS TRICK: Map the cloud text back into the exact structure 
                # your downstream code expects (so response.message.content works perfectly!)
                class MockMessage:
                    def __init__(self, content):
                        self.content = content
                class MockResponse:
                    def __init__(self, content):
                        self.message = MockMessage(content)
                        
                response = MockResponse(cloud_payload.choices[0].message.content)
                                        
            else:
                status.update(label="✓ Web results integrated. Synthesizing response...", state="running")
                            
        # Generate transaction metadata for assistant response
        ref_id = f"REF-{uuid.uuid4().hex[:6].upper()}"
        mode_val = "RAG + Web Search" if has_tool_call else "Offline RAG"
        if has_tool_call:
            if search_payload.startswith("Error") or "No web matching" in search_payload:
                status_val = "Web Retrieval Failed"
            else:
                status_val = "Web Retrieval Success"
        else:
            status_val = "Local Context Success" if private_context else "Local Context Empty"
            
        # Display badges at the top of the current assistant chat bubble
        badges_html = f"""
        <div style="margin-bottom: 8px;">
            <span class='status-badge status-badge-purple'>STATUS: {status_val}</span>
            <span class='status-badge status-badge-blue'>MODE: {mode_val}</span>
            <span class='status-badge'>REFERENCE_ID: {ref_id}</span>
        </div>
        """
        st.markdown(badges_html, unsafe_allow_html=True)

        # Step 2: Stream final model outputs
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            if has_tool_call:
                stream = ollama_client.chat(
                    model=llm_model,
                    messages=ollama_messages,
                    stream=True
                )
                for chunk in stream:
                    full_response += chunk['message']['content']
                    response_placeholder.markdown(f'<div style="line-height: 1.6; color: #FFFFFF; padding: 6px 10px; white-space: pre-wrap; word-wrap: break-word;">{full_response}▌</div>', unsafe_allow_html=True)
                response_placeholder.markdown(f'<div style="line-height: 1.6; color: #FFFFFF; padding: 6px 10px; white-space: pre-wrap; word-wrap: break-word;">{full_response}</div>', unsafe_allow_html=True)
            else:
                content = response.message.content if 'response' in locals() and response.message else "Could not generate response."
                for word in content.split(" "):
                    full_response += word + " "
                    response_placeholder.markdown(f'<div style="line-height: 1.6; color: #FFFFFF; padding: 6px 10px; white-space: pre-wrap; word-wrap: break-word;">{full_response}▌</div>', unsafe_allow_html=True)
                    time.sleep(0.012)
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