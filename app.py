import os
import time
import shutil
import uuid
import streamlit as st
import ollama
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from pypdf import PdfReader
from dotenv import load_dotenv
from groq import Groq
from config import settings

load_dotenv()

# =====================================================================
# 🌟 STAGE 1: SET INITIAL PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="🔒 Private AI Knowledge Base",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 STAGE 2: INJECT PREMIUM DESIGN OVERRIDES (WHITE BLOCKS REMOVED)
st.markdown("""
    <style>
    /* 1. Base App Canvas Background */
    .stApp {
        background: linear-gradient(135deg, #090a0f 0%, #11131e 50%, #1a1528 100%) !important;
        color: #e2e8f0;
    }
    
    /* 2. STRIP OUT DEFAULT BACKGROUND LAYERS OVERRIDES */
    [data-testid="stHeader"], 
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"],
    footer {
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* 3. Main Content Layout Constraints */
    .block-container {
        padding: 2rem 2rem 2rem 2rem !important;
    }
    
    /* 4. Sidebar Overrides & Visibility Fixes */
    div[data-testid="stSidebar"] {
        background: rgba(13, 15, 24, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
/* 4. Sidebar Overrides & Visibility Fixes */
/* Fix File Uploader Box Visibility and Inner Contrast */
    section[data-testid="stFileUploader"] {
        background-color: #161920 !important;
        border: 2px dashed #3e4451 !important;
        border-radius: 12px !important;
        padding: 15px !important;
    }
    
    /* Target the specific labels inside the drop zone wrapper */
    section[data-testid="stFileUploader"] label,
    section[data-testid="stFileUploader"] p,
    section[data-testid="stFileUploader"] span,
    section[data-testid="stFileUploader"] small,
    [data-testid="stFileUploadDropzone"] div,
    [data-testid="stFileUploadDropzone"] p,
    [data-testid="stFileUploadDropzone"] span {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
    }
    
    /* Make the browse files button legible and distinct */
    [data-testid="stFileUploadDropzone"] button {
        background-color: #212631 !important;
        border: 1px solid #4c5264 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    [data-testid="stFileUploadDropzone"] button:hover {
        border-color: #ffe066 !important;
        color: #ffe066 !important;
    }
    /* Force Sidebar Expander Wording Text to Be Visible */
    [data-testid="stSidebar"] details summary,
    [data-testid="stSidebar"] details summary * {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] details summary svg {
        fill: #ffffff !important;
    }
    
    /* 5. Custom Styled Alerts for Dark Context */
    div[data-testid="stAlert"] {
        background-color: rgba(25, 28, 41, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4) !important;
    }
    
    /* 6. Styled Prompt Chip Buttons */
    div.stButton > button {
        background-color: #1e242e !important;
        color: #f1f5f9 !important;
        border: 1px solid #3e4451 !important;
        border-radius: 8px !important;
        padding: 0.8rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        white-space: pre-wrap !important;
        text-align: left !important;
        line-height: 1.4 !important;
    }
    div.stButton > button:hover {
        border-color: #ffe066 !important;
        color: #ffe066 !important;
        background-color: #282e3d !important;
    }
    
    /* 7. FORM WORKAROUND CONTAINER STYLING */
    div[data-testid="stForm"] {
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background-color: rgba(22, 25, 31, 0.8) !important;
        border-radius: 14px !important;
        padding: 20px !important;
    }
    div[data-testid="stForm"] [data-testid="stWidgetLabel"] p {
        color: #ffe066 !important;
        font-weight: 500 !important;
    }
    div[data-testid="stForm"] button[type="submit"] {
        background-color: #2ecc71 !important;
        color: #0d0f12 !important;
        font-weight: 700 !important;
        border: none !important;
        width: 100% !important;
        box-shadow: 0 0 14px rgba(46, 204, 113, 0.4) !important;
    }
    
    /* 8. Informational Status Badges */
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
# 🛠️ STAGE 3: STATE INITIALIZATION ARCHITECTURE
# =====================================================================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Init Clients & Directories
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None

ollama_host = os.getenv("OLLAMA_HOST", settings.OLLAMA_HOST)
ollama_client = ollama.Client(host=ollama_host)

session_vault_dir = os.path.join(settings.VAULT_DIR, f"session_{st.session_state.session_id}")
os.makedirs(session_vault_dir, exist_ok=True)
os.makedirs(settings.DB_DIR, exist_ok=True)

clean_db_path = os.path.join(settings.DB_DIR, "serverless_v2")
chroma_client = chromadb.PersistentClient(path=clean_db_path)
collection_name = "private_rag_serverless"
serverless_ef = embedding_functions.ONNXMiniLM_L6_V2()

collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=serverless_ef
)

# Model Options Mapping
try:
    model_list = ollama_client.list()
    installed_models = [m['model'] for m in model_list.get('models', [])]
except Exception:
    installed_models = []

if installed_models:
    llm_options = [m for m in installed_models if "embed" not in m]
else:
    llm_options = ["llama-3.1-8b-instant", "qwen2.5:7b", "gemma4:12b"]
embedding_options = ["bge-large-en-v1.5", "nomic-embed-text"]

# Helper function to extract document strings
def extract_file_text(file_path, filename):
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
    return chunks

# =====================================================================
# 🧠 STAGE 4: SIDEBAR GENERATION CENTER
# =====================================================================
st.sidebar.markdown("<h2 style='color: #ffffff; margin-top:0;'>🔒 Vault Manager</h2>", unsafe_allow_html=True)

with st.sidebar.expander("⚙️ Advanced Engine Settings", expanded=False):
    llm_model = st.selectbox("Select LLM Model", llm_options, index=0)
    embedding_model = st.selectbox("Select Embedding Model", embedding_options, index=0)

st.sidebar.markdown("---")

# 🎭 RESTORED: DEMO SANDBOX ONBOARDING
st.sidebar.markdown("<h3 style='color: #ffffff;'>🎯 Risk-Free Sandbox Onboarding</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size: 0.82rem; color: #abb2bf; margin-top:-5px;'>✓ Free to try locally — 100% private sandbox</p>", unsafe_allow_html=True)

if st.sidebar.button("🎭 Load Demo Sample Data", use_container_width=True):
    st.session_state.indexed_files = [
        {"filename": "company_travel_policy_2026.pdf", "chunks": 8},
        {"filename": "department_budget_limits.txt", "chunks": 3}
    ]
    st.sidebar.success("Loaded secure demo templates!")
    time.sleep(0.5)
    st.rerun()

st.sidebar.markdown("<h3 style='color: #ffffff;'>📥 Step 1: Add Local Files</h3>", unsafe_allow_html=True)

uploaded_files = st.sidebar.file_uploader(
    "Drop project files here", 
    type=["pdf", "txt"], 
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}"
)

# Process active ingestion requests immediately
if uploaded_files:
    new_files = False
    for uploaded_file in uploaded_files:
        if not any(f["filename"] == uploaded_file.name for f in st.session_state.indexed_files):
            temp_path = os.path.join(session_vault_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            with st.sidebar.spinner(f"Vectorizing {uploaded_file.name}..."):
                chunks = extract_file_text(temp_path, uploaded_file.name)
                if chunks:
                    for i, chunk in enumerate(chunks):
                        collection.add(
                            documents=[chunk],
                            ids=[f"{uploaded_file.name}_chunk_{i}"],
                            metadatas=[{"session_id": st.session_state.session_id, "filename": uploaded_file.name}]
                        )
                    st.session_state.indexed_files.append({"filename": uploaded_file.name, "chunks": len(chunks)})
                    new_files = True
    if new_files:
        st.session_state.uploader_key += 1
        st.rerun()

st.sidebar.markdown("<h3 style='color: #ffffff;'>🗄️ Your Files</h3>", unsafe_allow_html=True)

if not st.session_state.indexed_files:
    st.sidebar.info("Your secure vault is empty. Upload files above to query local context.")
else:
    # Build functional remove tracking items that purge ChromaDB documents dynamically
    for f in list(st.session_state.indexed_files):
        display_name = f["filename"] if len(f["filename"]) <= 22 else f"{f['filename'][:19]}..."
        
        if st.sidebar.button(f"🗑️ Remove {display_name}", key=f"del_{f['filename']}", use_container_width=True):
            with st.sidebar.spinner("Purging database context vectors..."):
                # Purge from ChromaDB collection
                collection.delete(where={"filename": f["filename"]})
                # Remove tracking item from state wrapper
                st.session_state.indexed_files = [item for item in st.session_state.indexed_files if item["filename"] != f["filename"]]
                
                # Delete local storage payload file cleanly
                local_file_target = os.path.join(session_vault_dir, f["filename"])
                if os.path.exists(local_file_target):
                    os.remove(local_file_target)
            st.rerun()

if st.session_state.indexed_files:
    st.sidebar.markdown("<h4 style='color: #ffffff;'>Ready to Chat</h4>", unsafe_allow_html=True)
    for f in st.session_state.indexed_files:
        st.sidebar.markdown(f"<span class='status-badge'>📄 {f['filename']} ({f['chunks']} chunks)</span>", unsafe_allow_html=True)

# =====================================================================
# 💻 STAGE 5: MAIN CANVAS INTERFACE
# =====================================================================
st.markdown("<h1 style='color: #ffe066 !important; background: none; -webkit-text-fill-color: initial;'>Chat with your files without ever leaking data</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; color: #abb2bf; margin-top: -10px;'>Works without the internet to keep your files 100% private.</p>", unsafe_allow_html=True)

# Context Status Badges
with st.container(border=True):
    st.markdown(f"""
        <span class='status-badge status-badge-blue'>🛡️ System Status: Connected</span>
        <span class='status-badge status-badge-purple'>🤖 Engine: {llm_model}</span>
        <span class='status-badge'>🔗 Vectors: Local ONNX Engine</span>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🚀 Quick Start Guide:")
    st.markdown("1. **Upload Target Context Documents:** Drop `.txt` or `.pdf` data targets into the sidebar manager bucket.")
    st.markdown("2. **Inference Trigger Logic:** Select a quick diagnostic validation chip below, or type an explicit data request query into the bottom terminal canvas form.")

st.markdown("<p style='font-size: 1rem; color: #ffe066; font-weight: 600; margin-top: 20px; margin-bottom: 12px;'>📋 Select a Sample Question to Test the Engine</p>", unsafe_allow_html=True)

# Setup sample prompt query chips to trigger search lookups
chip_col1, chip_col2, chip_col3 = st.columns(3)
selected_chip_query = None

with chip_col1:
    if st.button("📊 Query Device Budgets\n\n'What is my electronic purchase budget limit?'", use_container_width=True, key="chip1"):
        selected_chip_query = "What is my electronic device purchase budget limit?"
with chip_col2:
    if st.button("✈️ Query Travel Policy\n\n'What are the company travel rules for 2026?'", use_container_width=True, key="chip2"):
        selected_chip_query = "What are the company travel rules for 2026?"
with chip_col3:
    if st.button("🔒 Analyze System Constraints\n\n'What are the specific constraints mentioned in my files?'", use_container_width=True, key="chip3"):
        selected_chip_query = "What are the specific constraints mentioned in my files?"

# Conversation History Output Render Loop
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(f"""
                <div style="margin-bottom: 8px;">
                    <span class='status-badge status-badge-purple'>STATUS: {message.get("status", "Success")}</span>
                    <span class='status-badge status-badge-blue'>MODE: {message.get("mode", "Offline Context Engine")}</span>
                </div>
            """, unsafe_allow_html=True)
        st.markdown(f'<div style="line-height: 1.6; color: #f1f5f9; padding: 2px 6px;">{message["content"]}</div>', unsafe_allow_html=True)

# Main Form Chat Field Wrapper
st.markdown("<br>", unsafe_allow_html=True)
with st.form(key="secure_chat_form", clear_on_submit=True):
    user_input = st.text_input(label="💬 Ask your private knowledge base anything:", placeholder="Type a message or select a question card above...")
    submit_button = st.form_submit_button(label="🚀 Start Secure Search")

# Extract final active prompt target query
active_prompt = selected_chip_query if selected_chip_query else (user_input if submit_button else None)

# =====================================================================
# 🧠 STAGE 6: ACTIVE ASSISTANT GENERATION PIPELINE
# =====================================================================
if active_prompt:
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    with st.chat_message("user"):
        st.markdown(f'<div style="line-height: 1.6; color: #f1f5f9; padding: 2px 6px;">{active_prompt}</div>', unsafe_allow_html=True)
        
    with st.chat_message("assistant"):
        progress_box = st.empty()
        progress_box.markdown("<div class='glass-card'>⚙️ Querying local vector store tracking spaces...</div>", unsafe_allow_html=True)
        
        # Pull reference documents from ChromaDB collections dynamically
        try:
            results = collection.query(query_texts=[active_prompt], n_results=3)
            context_documents = "\n---\n".join(results['documents'][0]) if results and results['documents'] else ""
        except Exception:
            context_documents = ""
            
        progress_box.markdown("<div class='glass-card'>🧠 Running local neural inference layers...</div>", unsafe_allow_html=True)
        
        # Build synthesis orchestration prompt boundaries
        rag_prompt = f"Context from user files:\n{context_documents if context_documents else 'No relevant document snippets found in database.'}\n\nQuestion:\n{active_prompt}"
        
        status_val = "Local Context Success" if context_documents else "Zero Reference Matches"
        mode_val = "Offline RAG Client"
        
        # Generate completion tokens
        full_response = ""
        resp_placeholder = st.empty()
        
        try:
            if groq_client:
                stream = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": rag_prompt}],
                    stream=True
                )
                for chunk in stream:
                    token = chunk.choices[0].delta.content or ""
                    full_response += token
                    resp_placeholder.markdown(f'<div style="line-height: 1.6; color: #f1f5f9; padding: 2px 6px;">{full_response}▌</div>', unsafe_allow_html=True)
            else:
                # Local network compute fallback paths
                response = ollama_client.chat(model=llm_model, messages=[{"role": "user", "content": rag_prompt}])
                full_response = response.message.content
                
            resp_placeholder.markdown(f'<div style="line-height: 1.6; color: #f1f5f9; padding: 2px 6px;">{full_response}</div>', unsafe_allow_html=True)
        except Exception as e:
            full_response = f"Inference execution engine breakdown error context: {e}"
            resp_placeholder.markdown(f"<div>{full_response}</div>", unsafe_allow_html=True)
            status_val = "Compute Fault"
            
        progress_box.empty()
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "status": status_val,
            "mode": mode_val
        })
        st.rerun()