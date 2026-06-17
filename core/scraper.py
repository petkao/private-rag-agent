from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

class HeadlessMarketScraper:
    @staticmethod
    def extract_clean_web_text(url: str) -> str:
        """Launches a localized sandboxed browser task to download layout text records cleanly."""
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, wait_until="networkidle", timeout=30000)
                html_raw = page.content()
                browser.close()
                
            soup = BeautifulSoup(html_raw, "html.parser")
            
            # Peel away heavy marketing layers, track vectors, CSS formats, and scripts
            for tags in soup(["script", "style", "nav", "footer", "header", "svg", "form"]):
                tags.extract()
                
            return soup.get_text(separator="\n", strip=True)
        except Exception as e:
            return f"Headless data retrieval bypass notice: {str(e)}"