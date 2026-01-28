"""
Multi-Agent AI System - Streamlit WORKING VERSION
Fixed async compatibility with LangGraph
"""

import os
import time
import random
import operator
from typing import List, Dict, Any, TypedDict, Annotated
from collections import Counter
import asyncio
import threading
from queue import Queue
from dotenv import load_dotenv
from datetime import datetime

from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

import streamlit as st

load_dotenv(dotenv_path="C:\AI_PROJECTS\Agentic-AI\.env", override=True)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Multi-Agent AI System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .thinking-log {
        background-color: #1e1e1e;
        color: #00ff00;
        padding: 1rem;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        overflow-x: auto;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .stButton>button {
        background-color: #667eea;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #764ba2;
    }
</style>
""", unsafe_allow_html=True)

# --- SYSTEM PROMPTS ---
CONSENSUS_SYSTEM_PROMPT = """You are part of a multi-agent expert panel. Your role is to provide the most accurate and precise answer possible based ONLY on the provided information.
EXTRACTION RULES:
- Parse all relevant data, names, and numbers from the search results.
- Cross-reference information from multiple sources to verify facts.
- Use contextual reasoning to resolve ambiguities.
- If the information is insufficient, state that clearly.

RESPONSE FORMAT: Always conclude with 'FINAL ANSWER: [PRECISE_ANSWER]'"""

REFLECTION_SYSTEM_PROMPT = """You are a reflection agent that validates answers from other agents.
Your task is to review the proposed answer based on the original question and the provided context.
1. Analyze the answer for relevance to the question.
2. Check for logical consistency and factual accuracy.
3. Verify the answer format is a direct and precise response.
4. Identify any obvious errors, hallucinations, or inconsistencies.

Respond with 'VALIDATED: [answer]' if it is correct, or 'CORRECTED: [better_answer]' if you can provide a more accurate or concise answer based on the context."""

# --- ASYNC HELPER FOR STREAMLIT ---
@st.cache_resource
def get_event_loop():
    """Create a dedicated event loop for async operations"""
    loop = asyncio.new_event_loop()
    
    def run_loop_forever(loop_to_run):
        asyncio.set_event_loop(loop_to_run)
        loop_to_run.run_forever()
    
    t = threading.Thread(target=run_loop_forever, args=(loop,), daemon=True)
    t.start()
    return loop

def run_async(coro):
    """Run async coroutine in Streamlit-compatible way"""
    loop = get_event_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()

# --- MODEL MANAGEMENT ---
class MultiModelManager:
    """Manages multiple Groq LLM models with current supported models"""
    def __init__(self):
        self.models = {}
        self._initialize_models()

    def _initialize_models(self):
        """Initialize available Groq models - UPDATED FOR JANUARY 2025"""
        groq_api_key = groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        
        if not groq_api_key:
            raise RuntimeError("❌ GROQ_API_KEY not found in environment variables")
        
        try:
            self.models['llama3.3_70b'] = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                api_key=groq_api_key
            )
            
            self.models['llama3.1_8b'] = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.1,
                api_key=groq_api_key
            )
            
            self.models['gpt-oss'] = ChatGroq(
                model="openai/gpt-oss-20b",
                temperature=0.1,
                api_key=groq_api_key
            )
            
            try:
                self.models['llama4_maverick'] = ChatGroq(
                    model="meta-llama/llama-4-maverick-17b-128e-instruct",
                    temperature=0.1,
                    api_key=groq_api_key
                )
            except:
                pass
            
            try:
                self.models['llama4_scout'] = ChatGroq(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    temperature=0.1,
                    api_key=groq_api_key
                )
            except:
                pass
            
            try:
                self.models['qwen3_32b'] = ChatGroq(
                    model="qwen/qwen3-32b",
                    temperature=0.1,
                    api_key=groq_api_key
                )
            except:
                pass
            
            if not self.models:
                raise RuntimeError("❌ No models could be initialized")
            
        except Exception as e:
            raise RuntimeError(f"❌ Failed to initialize Groq models: {e}")

    def get_diverse_models(self, count: int = 3) -> List:
        available_count = min(count, len(self.models))
        return random.sample(list(self.models.values()), available_count)

    def get_best_model(self) -> Any:
        for model_name in ['llama3.3_70b', 'llama4_maverick', 'qwen3_32b', 'llama4_scout', 'llama3.1_8b']:
            if model_name in self.models:
                return self.models[model_name]
        return list(self.models.values())[0] if self.models else None
    
    def get_model_names(self) -> List[str]:
        return list(self.models.keys())

# --- SEARCH TOOL ---
@tool
def enhanced_multi_search(query: str) -> str:
    """Search with Tavily and Wikipedia"""
    all_results = []
    
    if os.getenv("TAVILY_API_KEY"):
        try:
            search_tool = TavilySearchResults(max_results=3)
            results = search_tool.invoke(query)
            
            if isinstance(results, list):
                for doc in results:
                    if isinstance(doc, dict):
                        url = doc.get('url', 'N/A')
                        content = doc.get('content', '')
                        all_results.append(f"<WebResult url='{url}'>{content}</WebResult>")
                    elif isinstance(doc, str):
                        all_results.append(f"<WebResult>{doc}</WebResult>")
            elif isinstance(results, str):
                all_results.append(f"<WebResult>{results}</WebResult>")
        except Exception as e:
            pass
    
    try:
        docs = WikipediaLoader(query=query, load_max_docs=1).load()
        for doc in docs:
            content = doc.page_content[:1500]
            title = doc.metadata.get('title', 'Wikipedia')
            all_results.append(f"<WikiResult title='{title}'>{content}</WikiResult>")
    except:
        pass
    
    if not all_results:
        return "Search did not yield any results."
    
    return "\n\n---\n\n".join(all_results)

# --- CONSENSUS SYSTEM ---
class ConsensusVotingSystem:
    def __init__(self, model_manager: MultiModelManager):
        self.model_manager = model_manager
        self.reflection_agent = self._create_reflection_agent()

    def _create_reflection_agent(self):
        best_model = self.model_manager.get_best_model()
        return {'model': best_model, 'prompt': REFLECTION_SYSTEM_PROMPT} if best_model else None

    async def get_consensus_answer(self, query: str, search_results: str, thinking_log: List[str]) -> str:
        models = self.model_manager.get_diverse_models()
        if not models:
            thinking_log.append("❌ No models available")
            return "No models available"
        
        thinking_log.append(f"🤝 Starting consensus with {len(models)} agents...")
        tasks = [self._query_single_agent(model, query, search_results, i+1) for i, model in enumerate(models)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        valid_responses = [res for res in responses if isinstance(res, str) and "agent error" not in res.lower()]
        thinking_log.append(f"📝 Collected {len(valid_responses)} valid responses")

        if not valid_responses:
            thinking_log.append("❌ All agents failed")
            return "All agents failed to generate a response"
            
        consensus_answer = self._apply_consensus_voting(valid_responses, thinking_log)
        
        if self.reflection_agent:
            return await self._validate_with_reflection(consensus_answer, query, thinking_log)
        return consensus_answer

    async def _query_single_agent(self, model, query: str, search_results: str, agent_num: int) -> str:
        try:
            prompt = f"Question: {query}\n\nAvailable Information:\n{search_results}\n\nBased ONLY on the information above, provide the exact answer requested."
            response = await model.ainvoke([
                SystemMessage(content=CONSENSUS_SYSTEM_PROMPT), 
                HumanMessage(content=prompt)
            ])
            answer = response.content.strip()
            extracted = answer.split("FINAL ANSWER:")[-1].strip() if "FINAL ANSWER:" in answer else answer
            return extracted
        except Exception as e:
            return f"Agent error: {e}"

    def _apply_consensus_voting(self, responses: List[str], thinking_log: List[str]) -> str:
        thinking_log.append("🗳️  Voting on best answer...")
        if not responses:
            return "Unable to determine consensus"
        
        cleaned = [r.strip()[:500] for r in responses if r and len(r.strip()) > 10]
        
        if not cleaned:
            return "No valid responses to vote on"
        
        if len(set(r.lower()[:100] for r in cleaned)) == 1:
            thinking_log.append("✅ All agents agreed")
            return cleaned[0]
        
        vote_counts = Counter(r.lower()[:100] for r in cleaned)
        most_common_key = vote_counts.most_common(1)[0][0]
        
        for resp in cleaned:
            if resp.lower()[:100] == most_common_key:
                thinking_log.append(f"✅ Consensus reached")
                return resp
        
        return cleaned[0]

    async def _validate_with_reflection(self, answer: str, query: str, thinking_log: List[str]) -> str:
        if not self.reflection_agent:
            return answer
        
        thinking_log.append("🤔 Reflecting on answer...")
        try:
            prompt = f"Original Question: {query}\n\nProposed Answer: {answer}\n\nValidate this answer."
            response = await self.reflection_agent['model'].ainvoke([
                SystemMessage(content=self.reflection_agent['prompt']), 
                HumanMessage(content=prompt)
            ])
            result = response.content.strip()
            
            if "CORRECTED:" in result:
                thinking_log.append("📝 Answer was corrected")
                return result.split("CORRECTED:")[-1].strip()
            if "VALIDATED:" in result:
                thinking_log.append("✅ Answer validated")
                return result.split("VALIDATED:")[-1].strip()
            return answer
        except Exception as e:
            thinking_log.append(f"⚠️ Reflection failed: {str(e)[:50]}")
            return answer

# --- GRAPH SYSTEM ---
class AgentState(TypedDict):
    query: str
    search_results: str
    thinking_log: Annotated[List[str], operator.add]
    final_answer: str

class AutonomousLangGraphSystem:
    def __init__(self):
        self.model_manager = MultiModelManager()
        self.consensus_system = ConsensusVotingSystem(self.model_manager)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        g = StateGraph(AgentState)
        g.add_node("search", self._search_node)
        g.add_node("consensus", self._consensus_node)
        g.set_entry_point("search")
        g.add_edge("search", "consensus")
        g.add_edge("consensus", END)
        return g.compile()

    def _search_node(self, state: AgentState) -> dict:
        log = ["🔍 Searching multiple sources..."]
        search_results = enhanced_multi_search.invoke({"query": state["query"]})
        log.append("✅ Search complete")
        return {"search_results": search_results, "thinking_log": log}

    async def _consensus_node(self, state: AgentState) -> dict:
        log = []
        consensus_answer = await self.consensus_system.get_consensus_answer(
            state['query'], state['search_results'], log
        )
        return {"final_answer": consensus_answer, "thinking_log": log}
    
    def process_query(self, query: str) -> Dict:
        """FULLY FIXED: Use dedicated event loop in separate thread"""
        initial_state = {"query": query, "thinking_log": []}
        config = {"configurable": {"thread_id": f"agent_{time.time()}"}}
        try:
            # Use the Streamlit-compatible async runner
            final_state = run_async(self.graph.ainvoke(initial_state, config))
            
            return {
                "answer": final_state.get("final_answer", "Processing error"),
                "thinking_log": final_state.get("thinking_log", [])
            }
        except Exception as e:
            return {
                "answer": f"Error: {e}", 
                "thinking_log": [f"Error: {e}"]
            }

# --- SESSION STATE INITIALIZATION ---
def init_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'system' not in st.session_state:
        try:
            st.session_state.system = AutonomousLangGraphSystem()
            st.session_state.system_ready = True
        except Exception as e:
            st.session_state.system_ready = False
            st.session_state.error_message = str(e)
    if 'show_thinking' not in st.session_state:
        st.session_state.show_thinking = True

# --- STREAMLIT UI ---
def main():
    st.markdown('<h1 class="main-header">🤖 Multi-Agent AI System</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by Groq LLMs with Consensus Voting & Reflection")
    
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        if not st.session_state.system_ready:
            st.error("❌ System Error")
            st.error(st.session_state.get('error_message', 'Unknown error'))
            st.info("💡 Make sure GROQ_API_KEY and TAVILY_API_KEY are set in your .env file")
            return
        
        st.success("✅ System Ready")
        
        if st.session_state.system:
            model_names = st.session_state.system.model_manager.get_model_names()
            st.info(f"🤖 **Active Models:** {len(model_names)}")
            with st.expander("View Models"):
                for model in model_names:
                    st.text(f"• {model}")
        
        st.session_state.show_thinking = st.checkbox(
            "Show Thinking Process", 
            value=st.session_state.show_thinking,
            help="Display the internal reasoning of agents"
        )
        
        st.divider()
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        if st.session_state.messages:
            chat_export = "\n\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in st.session_state.messages
            ])
            st.download_button(
                label="📥 Download Chat",
                data=chat_export,
                file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        st.divider()
        st.caption("Built with Streamlit + LangGraph")
    
    st.markdown("---")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "thinking_log" in message and st.session_state.show_thinking:
                with st.expander("🧠 View Thinking Process"):
                    thinking_html = "<div class='thinking-log'>"
                    for log_entry in message["thinking_log"]:
                        thinking_html += f"{log_entry}<br>"
                    thinking_html += "</div>"
                    st.markdown(thinking_html, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Consulting agents..."):
                result = st.session_state.system.process_query(prompt)
                
                st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
                
                if st.session_state.show_thinking:
                    with st.expander("🧠 View Thinking Process"):
                        thinking_html = "<div class='thinking-log'>"
                        for log_entry in result["thinking_log"]:
                            thinking_html += f"{log_entry}<br>"
                        thinking_html += "</div>"
                        st.markdown(thinking_html, unsafe_allow_html=True)
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": result["answer"],
            "thinking_log": result["thinking_log"]
        })


if __name__ == "__main__":
    main()
