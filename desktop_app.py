import os
import sys
import shutil
from pypdf import PdfReader
import chromadb
import ollama
from duckduckgo_search import DDGS

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QTextEdit, QLineEdit, QPushButton)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QDragEnterEvent, QDropEvent

# ──────────────────────────────────────────────────────────────────────────────
# NATIVE PYTHON TOOLS FOR THE OLLAMA AGENT
# ──────────────────────────────────────────────────────────────────────────────

def web_search_tool(query: str) -> str:
    """
    Search the live web using DuckDuckGo to answer real-time, current, or general questions.
    Args:
        query: The search engine query string.
    Returns:
        str: A concatenated block of text containing search snippet records.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results:
                return "No web matching search indices returned from the query transaction."
            
            snippets = [f"Title: {r['title']}\nSource: {r['href']}\nContent: {r['body']}\n---" for r in results]
            return "\n".join(snippets)
    except Exception as e:
        return f"Error executing web search execution layer: {str(e)}"


class PrivateRagDesktopApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔒 Private Intel Vault — Multi-Modal Agent")
        self.setGeometry(100, 100, 720, 780)
        self.setStyleSheet("background-color: #1e222b; color: #abb2bf;")
        
        # 1. Establish System Paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.vault_dir = os.path.join(self.base_dir, "data", "vault")
        os.makedirs(self.vault_dir, exist_ok=True)
        
        # 2. Initialize ChromaDB Backend
        self.db_path = os.path.join(self.base_dir, "data", "db")
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="private_desktop_vault"
        )
        
        self.active_image_paths = []
        
        # 3. Build the UI
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Title Banner Layout
        title = QLabel("🔒 PRIVATE INTEL VAULT — AGENT LAYER")
        title.setFont(QFont("Helvetica", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #61afef; background-color: #282c34; padding: 12px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        # Drag & Drop Zone
        self.drop_zone = QLabel("📥 DRAG & DROP MULTIMODAL FILES HERE\n(Accepts: .pdf, .txt, .png, .jpg, .jpeg)")
        self.drop_zone.setFont(QFont("Helvetica", 11, QFont.Weight.Normal, True))
        self.drop_zone.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_zone.setStyleSheet("""
            QLabel {
                border: 2px dashed #abb2bf;
                border-radius: 5px;
                background-color: #2c313c;
                color: #abb2bf;
                padding: 30px;
            }
        """)
        self.setAcceptDrops(True)
        main_layout.addWidget(self.drop_zone)

        # Status Bar Monitoring Matrix
        self.status_label = QLabel("System Status: Operational | Tools Mounted: [Web Search] | Brain: Qwen 3 (8B)")
        self.status_label.setFont(QFont("Helvetica", 9))
        self.status_label.setStyleSheet("color: #98c379;")
        main_layout.addWidget(self.status_label)

        # Output Console Terminal Display
        self.output_text = QTextEdit()
        self.output_text.setFont(QFont("Courier", 10))
        self.output_text.setReadOnly(True)
        self.output_text.setStyleSheet("background-color: #21252b; color: #abb2bf; border: 1px solid #282c34;")
        main_layout.addWidget(self.output_text)
        self.log_output("✓ Native Agent Interface initialized flawlessly.\nDrop files to populate memory or type a query below.")

        # Input Prompt Layout Strip
        input_layout = QHBoxLayout()
        self.prompt_entry = QLineEdit()
        self.prompt_entry.setFont(QFont("Helvetica", 11))
        self.prompt_entry.setStyleSheet("background-color: #282c34; color: white; border: 1px solid #abb2bf; padding: 8px;")
        self.prompt_entry.returnPressed.connect(self.process_query)
        input_layout.addWidget(self.prompt_entry)

        submit_btn = QPushButton("Ask Agent")
        submit_btn.setFont(QFont("Helvetica", 10, QFont.Weight.Bold))
        submit_btn.setStyleSheet("background-color: #61afef; color: #1e222b; padding: 9px 15px; font-weight: bold;")
        submit_btn.clicked.connect(self.process_query)
        input_layout.addWidget(submit_btn)
        
        main_layout.addLayout(input_layout)

    # ──────────────────────────────────────────────────────────────────────────
    # DRAG & DROP ROUTING HANDLERS
    # ──────────────────────────────────────────────────────────────────────────
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_zone.setStyleSheet("border: 2px dashed #61afef; background-color: #21252b; color: #61afef; padding: 30px;")

    def dragLeaveEvent(self, event):
        self.drop_zone.setStyleSheet("border: 2px dashed #abb2bf; background-color: #2c313c; color: #abb2bf; padding: 30px;")

    def dropEvent(self, event: QDropEvent):
        self.drop_zone.setStyleSheet("border: 2px dashed #abb2bf; background-color: #2c313c; color: #abb2bf; padding: 30px;")
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if os.path.exists(file_path):
                self.handle_file(file_path)

    def handle_file(self, file_path):
        filename = os.path.basename(file_path)
        destination_path = os.path.join(self.vault_dir, filename)
        self.status_label.setText(f"Processing: {filename}...")
        self.status_label.setStyleSheet("color: #e5c07b;")
        QApplication.processEvents()
        
        try:
            shutil.copy(file_path, destination_path)
            
            if filename.lower().endswith('.pdf'):
                reader = PdfReader(destination_path)
                chunks = []
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
                
                for i, chunk in enumerate(chunks):
                    clean_text = chunk.encode('utf-8', errors='ignore').decode('utf-8').strip()
                    if clean_text:
                        response = ollama.embed(model="nomic-embed-text:latest", input=clean_text)
                        vector_embeddings = response['embeddings'][0]
                        self.collection.add(
                            ids=[f"{filename}_chunk_{i}"],
                            embeddings=[vector_embeddings],
                            documents=[clean_text],
                            metadatas=[{"format": "pdf", "path": destination_path}]
                        )
                self.log_output(f"✓ Indexed PDF in {len(chunks)} explicit vector chunks: {filename}")
                
            elif filename.lower().endswith('.txt'):
                with open(destination_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text_content = f.read()
                paragraphs = [p.strip() for p in text_content.split("\n\n") if p.strip()]
                for i, para in enumerate(paragraphs):
                    response = ollama.embed(model="nomic-embed-text:latest", input=para)
                    vector_embeddings = response['embeddings'][0]
                    self.collection.add(
                        ids=[f"{filename}_chunk_{i}"],
                        embeddings=[vector_embeddings],
                        documents=[para],
                        metadatas=[{"format": "txt", "path": destination_path}]
                    )
                self.log_output(f"✓ Indexed Text file in {len(paragraphs)} vector chunks: {filename}")
                
            elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_description = f"Local user data vault multi-modal vision tracking filename: {filename}"
                response = ollama.embed(model="nomic-embed-text:latest", input=image_description)
                vector_embeddings = response['embeddings'][0]
                self.collection.add(
                    ids=[filename],
                    embeddings=[vector_embeddings],
                    documents=[image_description],
                    metadatas=[{"format": "image", "path": destination_path}]
                )
                self.active_image_paths.append(destination_path)
                self.log_output(f"📸 Image explicitly loaded to Multimodal pipeline: {filename}")
                
            else:
                self.log_output(f"⚠️ Unsupported file type: {filename}")
                
            self.status_label.setText("System Status: Ready | Vault Sync Completed")
            self.status_label.setStyleSheet("color: #98c379;")
        except Exception as err:
            self.log_output(f"❌ Failed to process file: {str(err)}")

    # ──────────────────────────────────────────────────────────────────────────
    # ORCHESTRATION LAYER WITH NATIVE FUNCTION CALLING ROUTING
    # ──────────────────────────────────────────────────────────────────────────
    def process_query(self):
        query_text = self.prompt_entry.text().strip()
        if not query_text:
            return
            
        self.prompt_entry.clear()
        self.log_output(f"\n➔ Prompt Input Received: '{query_text}'\n🧠 Scanning local vector engine space...")
        QApplication.processEvents()
        
        try:
            # 1. Fetch relevant local device information context out of ChromaDB vector layers
            query_response = ollama.embed(model="nomic-embed-text:latest", input=query_text)
            query_vector = query_response['embeddings'][0]
            
            db_results = self.collection.query(query_embeddings=[query_vector], n_results=1)
            context_documents = db_results.get('documents', [[]])[0]
            context_metadata = db_results.get('metadatas', [[]])[0]
            
            private_context = context_documents[0] if context_documents else "No matching private device documentation parameters established."
            
            # Setup image lists for vision-processing elements
            images_to_evaluate = []
            if context_metadata:
                for meta in context_metadata:
                    if meta.get('format') == 'image' and os.path.exists(meta.get('path')):
                        images_to_evaluate.append(meta.get('path'))
            for img_path in self.active_image_paths:
                if img_path not in images_to_evaluate:
                    images_to_evaluate.append(img_path)

            # 2. Send initial message array to Qwen 3 containing our explicit search tool blueprint
            self.log_output("🤖 Evaluation step: Checking if live web data is required...")
            QApplication.processEvents()
            
            system_instructions = (
                "You are an advanced local AI agent with access to an internal user database vault AND a live web search tool.\n"
                f"Here is what was found in the local device data vault for this query: '{private_context}'.\n\n"
                "CRITICAL INSTRUCTION: Analyze the user's question. If the local data vault answers it completely, "
                "do NOT use the web search tool. If the question requires real-time facts, current dates, things outside "
                "the local documents, or the local data is insufficient, call the 'web_search_tool' function."
            )
            
            messages = [
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": query_text}
            ]
            
            # First chat call passes the actual function reference into the tools collection array
            response = ollama.chat(
                model="qwen3:8b",
                messages=messages,
                tools=[web_search_tool]
            )
            
            # 3. Check if the model decided it needed to call the search engine tool
            if response.message.tool_calls:
                for tool in response.message.tool_calls:
                    if tool.function.name == "web_search_tool":
                        search_args = tool.function.arguments
                        search_query = search_args.get("query", query_text)
                        
                        self.log_output(f"🌐 Agent executed Web Search Tool wrapper with query: '{search_query}'")
                        QApplication.processEvents()
                        
                        # Execute the physical python duckduckgo scraper function
                        search_payload = web_search_tool(query=search_query)
                        
                        # Add the model's intent request and the actual search results back to the message history logs
                        messages.append(response.message)
                        messages.append({
                            "role": "tool",
                            "tool_name": "web_search_tool",
                            "content": search_payload
                        })
                        
                        self.log_output("📥 Web payload integrated. Compiling final synthesis...")
                        QApplication.processEvents()
                        
                        # Re-invoke Ollama to review the fresh live internet contents
                        response = ollama.chat(model="qwen3:8b", messages=messages)
            
            # 4. Final Output Render Loop Stream
            agent_output = response['message']['content']
            self.log_output(f"\n=================== SYSTEM AGENT PAYLOAD ANALYSIS ===================\n{agent_output}\n=====================================================================")
            self.active_image_paths.clear()
            
        except Exception as err:
            self.log_output(f"\n❌ Operational failure executing agent core loop: {str(err)}")

    def log_output(self, text):
        self.output_text.append(text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PrivateRagDesktopApp()
    window.show()
    sys.exit(app.exec())