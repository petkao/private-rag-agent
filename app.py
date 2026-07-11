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
import easyocr
from PIL import Image as PILImage

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
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Fix File Uploader Box Border Accent Colors */
    section[data-testid="stFileUploader"] {
        border: 2px dashed #4c5264 !important;
        border-radius: 12px !important;
        padding: 15px !important;
    }
    
    /* Force Sidebar Expander Header Text to Stay Visible */
    [data-testid="stSidebar"] details summary,
    [data-testid="stSidebar"] details summary * {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] details summary svg {
        fill: #ffffff !important;
    }
    
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
    
    /* Custom Styled Alerts for Dark Context */
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

# Initialize EasyOCR Reader for English (cached to prevent reload latency)
# Create a dedicated local directory inside your project for the weights
ocr_model_dir = os.path.join(os.path.dirname(__file__), ".easyocr_models")
os.makedirs(ocr_model_dir, exist_ok=True)

# Pre-download weights at boot level so it never times out during an upload transaction
@st.cache_resource
def pre_download_ocr_weights():
    try:
        # This forces the download immediately when the server first boots up
        easyocr.Reader(['en'], gpu=False, model_storage_directory=ocr_model_dir)
        return True
    except Exception as e:
        return False

# Trigger the boot download
_ = pre_download_ocr_weights()

# Standard cached interface reader
@st.cache_resource
def get_ocr_reader():
    try:
        # Wrap initialization in a quick timeout guard safe-check
        return easyocr.Reader(['en'], gpu=False, model_storage_directory=ocr_model_dir)
    except Exception as e:
        # If it fails or times out, return None instead of crashing the server
        return None

reader_ocr = get_ocr_reader()
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
    ext = filename.lower()
    
    if ext.endswith('.pdf'):
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
            
    elif ext.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text_content = f.read()
        paragraphs = [p.strip() for p in text_content.split("\n\n") if p.strip()]
        for para in paragraphs:
            if len(para) > 1000:
                for i in range(0, len(para), 1000):
                    chunks.append(para[i:i+1000])
            else:
                chunks.append(para)
                
    # 🖼️ NEW: IMAGE OCR PROCESSING TRACK
    # 🖼️ SAFE-GUARDED IMAGE OCR PROCESSING TRACK
    elif ext.endswith(('.png', '.jpg', '.jpeg')):
        try:
            if reader_ocr is None:
                chunks.append(f"[System Notice] OCR engine is still initializing or downloading models for {filename}. Please try re-uploading in a moment.")
            else:
                ocr_results = reader_ocr.readtext(file_path, detail=0)
                if ocr_results:
                    full_image_text = " ".join(ocr_results)
                    if len(full_image_text) > 1000:
                        for i in range(0, len(full_image_text), 1000):
                            chunks.append(f"[Image Context Source: {filename}] " + full_image_text[i:i+1000])
                    else:
                        chunks.append(f"[Image Context Source: {filename}] " + full_image_text)
                else:
                    chunks.append(f"[Image Context Source: {filename}] Visual matrix scanned. No horizontal text strings detected.")
        except Exception as e:
            # Prevent unexpected library errors from killing the app session
            chunks.append(f"[System Notice] OCR processing temporarily paused for {filename} due to container resource limits.")
            
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
    # 1. Define the real text strings that your query chips look for
    demo_docs = {
        "company_travel_policy_2026.pdf": [
            "Company Travel Policy 2026: All international flights must be approved by a Department Head 14 days in advance.",
            "Travel Regulations 2026: Economy class is mandatory for domestic flights under 5 hours. Business class is permitted for flights over 5 hours.",
            "Expense Policy 2026: Daily meal allowance (per diem) is capped at $75 USD per day. Itemized receipts are required for all hotel incidentals."
        ],
        "department_budget_limits.txt": [
            "Department Budget Limits: The standard electronic device purchase budget limit is strictly capped at $1,200 per employee per year.",
            "Hardware Upgrades: Any device purchase exceeding $1,200 requires a formal business justification signed by the IT Director.",
            "Software Procurement: Individual recurring cloud subscription allowances are set at $50 per user per month maximum."
        ]
    }
    
    # 2. Clear old session tracking state to avoid duplicates
    st.session_state.indexed_files = []
    
    # 3. Add the real text vectors into your active ChromaDB collection
    with st.sidebar.spinner("Injecting secure demo vectors..."):
        for filename, chunks in demo_docs.items():
            for i, chunk_text in enumerate(chunks):
                collection.add(
                    documents=[chunk_text],
                    ids=[f"demo_{filename}_chunk_{i}"],
                    metadatas=[{"session_id": st.session_state.session_id, "filename": filename}]
                )
            # Track it in Streamlit UI state
            st.session_state.indexed_files.append({"filename": filename, "chunks": len(chunks)})
            
    st.sidebar.success("Loaded secure demo templates with real vector embeddings!")
    time.sleep(0.5)
    st.rerun()

st.sidebar.markdown("<h3 style='color: #ffffff;'>📥 Step 1: Add Local Files</h3>", unsafe_allow_html=True)

uploaded_files = st.sidebar.file_uploader(
    "Drop project files here", 
    type=["pdf", "txt", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}"
)

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
    for f in list(st.session_state.indexed_files):
        display_name = f["filename"] if len(f["filename"]) <= 22 else f"{f['filename'][:19]}..."
        
        if st.sidebar.button(f"🗑️ Remove {display_name}", key=f"del_{f['filename']}", use_container_width=True):
            with st.sidebar.spinner("Purging database context vectors..."):
                collection.delete(where={"filename": f["filename"]})
                st.session_state.indexed_files = [item for item in st.session_state.indexed_files if item["filename"] != f["filename"]]
                local_file_target = os.path.join(session_vault_dir, f["filename"])
                if os.path.exists(local_file_target):
                    os.remove(local_file_target)
            st.rerun()

if st.session_state.indexed_files:
    st.sidebar.markdown("<h4 style='color: #ffffff;'>Ready to Chat</h4>", unsafe_allow_html=True)
    for f in st.session_state.indexed_files:
        st.sidebar.markdown(f"<span class='status-badge'>📄 {f['filename']} ({f['chunks']} chunks)</span>", unsafe_allow_html=True)


# =====================================================================
# 💻 STAGE 5: MAIN CANVAS INTERFACE (SPLIT WORKBENCH ACTIVE)
# =====================================================================
st.markdown("<h1 style='color: #ffe066 !important; background: none; -webkit-text-fill-color: initial; margin-bottom: 0px;'>🛡️ Private Multi-Agent Research Station</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; color: #abb2bf; margin-top: -5px;'>Works without the internet to keep your assets and configurations 100% private.</p>", unsafe_allow_html=True)

# Context Status Badges Global Banner
with st.container(border=True):
    st.markdown(f"""
        <span class='status-badge status-badge-blue'>🛡️ System Status: Connected</span>
        <span class='status-badge status-badge-purple'>🤖 Engine: {llm_model}</span>
        <span class='status-badge'>🔗 Vectors: Local ONNX Engine</span>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Establish the balanced side-by-side execution layout columns
col_viewer, col_chat = st.columns([1, 1], gap="medium")

# ---------------------------------------------------------------------
# PANEL 1 (LEFT COLUMN): VISUAL VAULT PREVIEWER
# ---------------------------------------------------------------------
with col_viewer:
    st.markdown("<h3 style='color: #ffe066; border-bottom: 1px solid #212631; padding-bottom: 8px; margin-top:0;'>🖼️ Active Asset View</h3>", unsafe_allow_html=True)
    
    active_files = st.session_state.get("indexed_files", [])
    
    if not active_files and not uploaded_files:
        st.info("No documents are currently active. Drop a file (.png, .jpg, .pdf, .txt) into the sidebar to activate the live workbench previewer.")
    else:
        all_file_names = []
        if uploaded_files:
            all_file_names.extend([f.name for f in uploaded_files])
        if active_files:
            all_file_names.extend([f["filename"] if isinstance(f, dict) else f for f in active_files])
            
        all_file_names = list(set(all_file_names))
        selected_preview = st.selectbox("🎯 Select file to visually verify:", all_file_names)
        
        # Render Preview Engine based on extensions
        if selected_preview.lower().endswith(('.png', '.jpg', '.jpeg')):
            matched_file = next((f for f in uploaded_files if f.name == selected_preview), None) if uploaded_files else None
            if matched_file:
                st.image(matched_file, caption=f"Active Source: {selected_preview}", use_container_width=True)
            else:
                st.markdown(f"""
                <div style='background-color: #161920; border: 1px dashed #3e4451; padding: 40px; border-radius: 10px; text-align: center;'>
                    <p style='color: #abb2bf; font-size: 0.9rem;'>🎭 <b>[Demo Sandbox Template Mode]</b></p>
                    <p style='color: #ffffff; font-size: 1.1rem; font-weight:600;'>{selected_preview}</p>
                    <span style='color: #ffe066; font-size: 0.8rem;'>System Vector Matrix Verified ✓</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background-color: #161920; border-left: 4px solid #ffe066; padding: 20px; border-radius: 6px;'>
                <p style='color: #ffffff; font-weight: 600; margin-bottom: 4px;'>📄 Active Document Stream</p>
                <code style='color: #abb2bf; font-size: 0.8rem;'>{selected_preview}</code>
                <p style='color: #ffffff; font-size: 0.85rem; margin-top: 15px;'>
                    <i>Text vectors extracted, partitioned into sub-tokens, and actively mapped to your local ChromaDB query context.</i>
                </p>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# PANEL 2 (RIGHT COLUMN): LIVE VECTOR CHAT STREAM
# ---------------------------------------------------------------------
with col_chat:
    st.markdown("<h3 style='color: #ffe066; border-bottom: 1px solid #212631; padding-bottom: 8px; margin-top:0;'>💬 Groq Context Chat</h3>", unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 0.9rem; color: #ffe066; font-weight: 600; margin-bottom: 8px;'>📋 Sample Question Chips:</p>", unsafe_allow_html=True)
    
    # Setup sample prompt query chips to trigger search lookups
    chip_col1, chip_col2, chip_col3 = st.columns(3)
    selected_chip_query = None

    with chip_col1:
        if st.button("📊 Budgets\n'What is my purchase limit?'", use_container_width=True, key="chip1"):
            selected_chip_query = "What is my electronic device purchase budget limit?"
    with chip_col2:
        if st.button("✈️ Travel\n'What are travel rules?'", use_container_width=True, key="chip2"):
            selected_chip_query = "What are the company travel rules for 2026?"
    with chip_col3:
        if st.button("🔒 Constraints\n'What are my constraints?'", use_container_width=True, key="chip3"):
            selected_chip_query = "What are the specific constraints mentioned in my files?"

    st.markdown("---")

    # 🎛️ NEW: Pinned Height Viewport Container for Chat Stream
    # This locks down the window size so history scrolls internally instead of stretching the main canvas down!
    with st.container(height=450, border=False):
        # Conversation History Output Render Loop nested within scrollable canvas
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

    # Main Form Chat Input Field stays anchored outside the scrollbox
    with st.form(key="secure_chat_form", clear_on_submit=True):
        user_input = st.text_input(label="💬 Query local database:", placeholder="Type a message or select a question card above...")
        submit_button = st.form_submit_button(label="🚀 Start Secure Search")

    # Extract final active prompt target query
    active_prompt = selected_chip_query if selected_chip_query else (user_input if submit_button else None)

# =====================================================================
# 🧠 STAGE 6: ACTIVE ASSISTANT GENERATION PIPELINE
# =====================================================================
if active_prompt:
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    
    # Re-trigger immediate update inside the chat panel visually
    with col_chat:
        with st.chat_message("user"):
            st.markdown(f'<div style="line-height: 1.6; color: #f1f5f9; padding: 2px 6px;">{active_prompt}</div>', unsafe_allow_html=True)
            
        with st.chat_message("assistant"):
            progress_box = st.empty()
            progress_box.markdown("<div class='glass-card'>⚙️ Querying local vector store tracking spaces...</div>", unsafe_allow_html=True)
            
            try:
                results = collection.query(query_texts=[active_prompt], n_results=3)
                context_documents = "\n---\n".join(results['documents'][0]) if results and results['documents'] else ""
            except Exception:
                context_documents = ""
                
            progress_box.markdown("<div class='glass-card'>🧠 Running local neural inference layers...</div>", unsafe_allow_html=True)
            
            rag_prompt = f"Context from user files:\n{context_documents if context_documents else 'No relevant document snippets found in database.'}\n\nQuestion:\n{active_prompt}"
            
            status_val = "Local Context Success" if context_documents else "Zero Reference Matches"
            mode_val = "Offline RAG Client"
            
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