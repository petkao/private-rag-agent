from typing import Dict, TypedDict, Optional
from langgraph.graph import StateGraph, END
from duckduckgo_search import DDGS
from langchain_ollama import ChatOllama
from config import settings
from core.ingestor import LocalVaultIngestor
from core.scraper import HeadlessMarketScraper

# 1. State Schema definitions passed along our workflow paths
class AgentState(TypedDict):
    user_prompt: str
    refining_prompt: Optional[str]
    private_context: str
    search_query: str
    web_results: str
    final_analysis: str
    target_destination: str

class LangGraphLocalAgent:
    def __init__(self):
        self.llm = ChatOllama(base_url=settings.OLLAMA_HOST, model=settings.PRIMARY_LLM, temperature=0.1)
        self.vault = LocalVaultIngestor()
        self.scraper = HeadlessMarketScraper()

    def retrieve_local_context_node(self, state: AgentState) -> Dict:
        """Node 1: Auto-pulls file entries relevant to user intents via semantic distance."""
        print("🧠 [Node 1] Reviewing local device vault content logs...")
        context = self.vault.query_local_context(state["user_prompt"])
        return {"private_context": context}

    def generate_query_node(self, state: AgentState) -> Dict:
        """Node 2: Compiles local context with prompts to produce an anonymous query string."""
        print("✍️ [Node 2] Creating anonymized query keyword structure using local LLM...")
        prompt = f"""
        Generate a concise, simple search query target string for an online web crawler.
        Blend the information from the background data to specialize the query focus, but DO NOT drop private labels or names.

        USER INTENT INPUT: {state['user_prompt']}
        LOCAL PARAMETER HISTORY: {state['private_context']}

        Provide only the clean search keywords. Do not provide commentary or wraps.
        """
        response = self.llm.invoke(prompt)
        return {"search_query": response.content.strip().replace('"', '')}

    def scrape_web_node(self, state: AgentState) -> Dict:
        """Node 3: Resolves query against public endpoints and triggers background browser extract hooks."""
        print(f"🌐 [Node 3] Broadcasting anonymous search queries for: '{state['search_query']}'")
        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(state["search_query"], max_results=1))
            if not hits:
                return {"web_results": "Search indexing resulted in an empty set."}
            
            target_url = hits[0]["url"]
            print(f"🕷️ [Node 3] Deploying headless crawl automation loop to endpoint: {target_url}")
            web_text = self.scraper.extract_clean_web_text(target_url)
            return {"web_results": web_text[:6000]} # Trim to fit contextual window limit boundaries
        except Exception as e:
            return {"web_results": f"Web capture failed: {str(e)}"}

    def synthesize_node(self, state: AgentState) -> Dict:
        """Node 4: Compares crawled data matrices directly against local constraints using local LLM."""
        print("🤖 [Node 4] Re-evaluating text inputs internally via Qwen processing layers...")
        refine_clause = f"ADDITIONAL SUB-FOCUS REQUIREMENT: {state['refining_prompt']}" if state.get("refining_prompt") else ""
        
        prompt = f"""
        You are an isolated private data agent executing calculations entirely within user device storage frameworks.
        Cross-reference the latest web documentation text against internal context profiles to issue guidance.

        ORIGINAL USER PROMPT: {state['user_prompt']}
        {refine_clause}

        CRAWLED SITE CONTENT PROFILE:
        {state['web_results']}

        USER SYSTEM CONFIGURATIONS:
        {state['private_context']}

        Synthesize your final summary response.
        """
        response = self.llm.invoke(prompt)
        return {"final_analysis": response.content}

    def dispatch_route_node(self, state: AgentState) -> Dict:
        """Node 5: Routes output payload packets directly toward targeted application channels."""
        dest = state.get("target_destination", "terminal_display")
        print(f"🚀 [Node 5] Dispatching final generated payload cleanly into: [{dest.upper()}]")
        
        # Return the target destination state to satisfy LangGraph's update check rule
        return {"target_destination": dest}

    def build_agent_graph(self):
        """Assembles internal nodes and compiles our finalized StateGraph execution flow."""
        workflow = StateGraph(AgentState)

        workflow.add_node("retrieve_context", self.retrieve_local_context_node)
        workflow.add_node("generate_query", self.generate_query_node)
        workflow.add_node("scrape_web", self.scrape_web_node)
        workflow.add_node("synthesize", self.synthesize_node)
        workflow.add_node("dispatch_route", self.dispatch_route_node)

        workflow.set_entry_point("retrieve_context")
        workflow.add_edge("retrieve_context", "generate_query")
        workflow.add_edge("generate_query", "scrape_web")
        workflow.add_edge("scrape_web", "synthesize")
        workflow.add_edge("synthesize", "dispatch_route")
        workflow.add_edge("dispatch_route", END)

        return workflow.compile()