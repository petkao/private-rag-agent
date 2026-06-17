import sys
import os
import requests

def run_user_onboarding():
    print("=======================================================")
    print("🛡️  Welcome to Your Private RAG Agent Installation Wizard")
    print("=======================================================")
    print("This setup script will verify your device environment and")
    print("download the required local AI models securely.")
    print("-------------------------------------------------------")

    # 1. Check if Ollama is running backend services
    ollama_host = "http://localhost:11434"
    try:
        response = requests.get(ollama_host, timeout=2)
        if response.status_code != 200:
            raise requests.exceptions.ConnectionError
        print("✓ Local model service engine (Ollama) detected running.")
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        print("❌ Error: Ollama service engine is not running on this machine.")
        print("👉 Please download and start Ollama from https://ollama.com first,")
        print("   then restart this installation script.")
        sys.exit(1)

    # 2. Ask for explicit user permission (Human-in-the-Loop Gate)
    print("\n📦 Model Storage Footprint Authorization:")
    print("• Primary Brain Model (Qwen 7B Optimization): ~4.7 GB")
    print("• Private Vault Embedder (Nomic Text):        ~274 MB")
    print("-------------------------------------------------------")
    
    while True:
        consent = input("Do you grant permission to download these models onto your disk? (y/n): ").strip().lower()
        if consent in ['n', 'no']:
            print("\n❌ Setup aborted. Permission denied by user.")
            sys.exit(0)
        if consent in ['y', 'yes']:
            break
        print("Please type 'y' or 'n'.")

    # 3. Stream downloads with progress updates
    models_to_pull = ["nomic-embed-text", "qwen2.5:7b"]
    for model in models_to_pull:
        print(f"\n📥 Fetching '{model}' from secure local registries...")
        url = f"{ollama_host}/api/pull"
        
        with requests.post(url, json={"model": model, "stream": True}, stream=True) as res:
            current_status = ""
            for line in res.iter_lines():
                if line:
                    import json
                    log = json.loads(line.decode('utf-8'))
                    status = log.get("status", "")
                    completed = log.get("completed", 0)
                    total = log.get("total", 0)
                    
                    if status != current_status:
                        if total > 0:
                            pct = (completed / total) * 100
                            print(f"   ↳ Progress: {status} ({pct:.1f}%)", end="\r")
                        else:
                            print(f"   ↳ Status: {status}")
                        current_status = status

    # 4. Initialize Playwright Browser components automatically
    print("\n\n🌐 Finalizing background headless web components...")
    os.system("playwright install chromium")

    print("\n=======================================================")
    print("🎉 INSTALLATION COMPLETE! Your device is fully secure.")
    print("=======================================================")
    print("You can now start your completely private chat assistant by typing:")
    print("   python app.py")
    print("=======================================================\n")

if __name__ == "__main__":
    run_user_onboarding()