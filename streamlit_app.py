"""
Agentic AI System - Streamlit Interface
Multi-agent consensus system with beautiful UI
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="Agentic AI System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .thinking-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .answer-box {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        font-size: 1.2rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize system
@st.cache_resource
def initialize_system():
    """Initialize the AI system once and cache it"""
    try:
        from main import AutonomousLangGraphSystem
        return AutonomousLangGraphSystem()
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        return None

# Header
st.markdown('<p class="main-header">🤖 Agentic AI System</p>', unsafe_allow_html=True)
st.markdown("### Multi-Agent Consensus System with Transparent Thinking")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This system uses **multiple AI agents** that:
    - 🔍 Search the web for information
    - 🤝 Vote on the best answer
    - 🧠 Reflect and validate results
    
    **Powered by:**
    - 3x Groq AI Models
    - Tavily Search
    - Wikipedia
    """)
    
    st.divider()
    
    st.header("🎯 How It Works")
    st.markdown("""
    1. **Search Phase**: Gather information from multiple sources
    2. **Consensus Phase**: 3 AI agents analyze and vote
    3. **Reflection Phase**: Validate and refine the answer
    """)
    
    st.divider()
    
    st.header("📊 System Status")
    system = initialize_system()
    if system:
        st.success(f"✅ System Ready")
        st.info(f"🤖 {len(system.model_manager.models)} models active")
    else:
        st.error("❌ System Not Ready")
    
    st.divider()
    
    st.markdown("**Example Questions:**")
    st.markdown("""
    - What is quantum computing?
    - Who won the 2024 Nobel Prize in Physics?
    - Explain machine learning in simple terms
    - What are the benefits of renewable energy?
    """)

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    # Query input
    query = st.text_area(
        "💬 Ask me anything:",
        height=100,
        placeholder="Type your question here... (e.g., 'What is artificial intelligence?')"
    )

with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    process_button = st.button("🚀 Process Query", type="primary", use_container_width=True)
    clear_button = st.button("🗑️ Clear", use_container_width=True)

if clear_button:
    st.rerun()

# Process query
if process_button and query:
    system = initialize_system()
    
    if not system:
        st.error("❌ System is not initialized. Please check your API keys in Streamlit secrets.")
        st.info("Set GROQ_API_KEY and TAVILY_API_KEY in Settings → Secrets")
        st.stop()
    
    with st.spinner("🤔 Processing your query..."):
        try:
            result = system.process_query(query)
            
            # Display answer
            st.markdown("---")
            st.markdown("### 💡 Answer")
            st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
            
            # Display thinking process
            st.markdown("---")
            st.markdown("### 🧠 Thinking Process")
            
            with st.expander("Click to see detailed thinking log", expanded=True):
                for i, log_entry in enumerate(result.get("thinking_log", [])):
                    st.markdown(f'<div class="thinking-box">{log_entry}</div>', unsafe_allow_html=True)
            
            # Stats
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Models Used", len(system.model_manager.models))
            with col2:
                st.metric("Thinking Steps", len(result.get("thinking_log", [])))
            with col3:
                st.metric("Status", "✅ Success")
                
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            st.exception(e)

elif process_button and not query:
    st.warning("⚠️ Please enter a question first!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>Built with ❤️ using Streamlit, LangChain, and Groq AI</p>
    <p>🔒 Your queries are processed securely and not stored</p>
</div>
""", unsafe_allow_html=True)
