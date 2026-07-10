# theme.py
import streamlit as st

def inject_custom_theme():
    """Injects premium dark workspace layout layers and UI styling."""
    st.markdown("""
        <style>
            .stApp {
                background-color: #0d0f12 !important;
                color: #e2e8f0 !important;
                font-family: -apple-system, BlinkMacSystemFont, sans-serif !important;
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
            [data-testid="stChatInput"] {
                background-color: #16191f !important;
                border: 1px solid #262c36 !important;
                border-radius: 12px !important;
                color: #ffffff !important;
            }
            div[data-testid="stColumn"] button {
                background-color: #16191f !important;
                border: 1px solid #262c36 !important;
                border-radius: 20px !important;
                color: #94a3b8 !important;
                font-size: 0.82rem !important;
                transition: all 0.2s ease-in-out !important;
            }
            div[data-testid="stColumn"] button:hover {
                border-color: #60a5fa !important;
                color: #ffffff;
            }
        </style>
    """, unsafe_allow_html=True)

def render_suggestion_label():
    """Renders minimalist typography for prompt templates."""
    st.markdown("<p style='font-size: 0.85rem; color: #64748b; margin-top: 1rem; margin-bottom: 0.25rem;'>💡 Quick start templates:</p>", unsafe_allow_html=True)