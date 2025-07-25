"""
Ultra-Enhanced Multi-Agent LLM System with Consensus Voting
Implements latest 2024-2025 research for maximum evaluation performance.
This version is fully autonomous and contains no pre-programmed knowledge.
"""

import os
import time
import random
import operator
import re
from typing import List, Dict, Any, TypedDict, Annotated
from dotenv import load_dotenv
from collections import Counter
import asyncio
import nest_asyncio

# Apply the patch to allow nested event loops in environments like Jupyter
nest_asyncio.apply()

from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq

# Open-source model integrations
try:
    from langchain_ollama import ChatOllama
    from langchain_together import ChatTogether
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

load_dotenv()

# --- GENERALIZED SYSTEM PROMPTS ---

CONSENSUS_SYSTEM_PROMPT = """You are part of a multi-agent expert panel. Your role is to provide the most accurate and precise answer possible based *only* on the provided information.
    EXTRACTION RULES:
- Parse all relevant data, names, and numbers from the search results.
- Cross-reference information from multiple sources to verify facts.
- Use contextual reasoning to resolve ambiguities.
- If the information is insufficient to answer the question, state that clearly.

RESPONSE FORMAT: Always conclude with 'FINAL ANSWER: [PRECISE_ANSWER]'"""

REFLECTION_SYSTEM_PROMPT = """You are a reflection agent that validates answers from other agents.
Your task is to review the proposed answer based on the original question and the provided context.
1.  Analyze the answer for relevance to the question.
2.  Check for logical consistency and factual accuracy.
3.  Verify the answer format is a direct and precise response.
4.  Identify any obvious errors, hallucinations, or inconsistencies.

Respond with 'VALIDATED: [answer]' if it is correct, or 'CORRECTED: [better_answer]' if you can provide a more accurate or concise answer based on the context."""

# --- MODEL AND TOOL MANAGEMENT ---

class MultiModelManager:
    """Manages multiple open-source and commercial LLM models"""
    def __init__(self):
        self.models = {}
        self._initialize_models()

    def _initialize_models(self):
        """Initialize available models in priority order"""
        if os.getenv("GROQ_API_KEY"):
            self.models['groq_llama3_70b'] = ChatGroq(model="llama3-70b-8192", temperature=0.1, api_key=os.getenv("GROQ_API_KEY"))
        if OLLAMA_AVAILABLE:
            try:
                self.models['ollama_llama3'] = ChatOllama(model="llama3")
            except Exception as e:
                print(f"Ollama models not available: {e}")
        if os.getenv("TOGETHER_API_KEY"):
            try:
                self.models['together_llama3'] = ChatTogether(model="meta-llama/Llama-3-70b-chat-hf", api_key=os.getenv("TOGETHER_API_KEY"))
            except Exception as e:
                print(f"Together AI models not available: {e}")
        print(f"✅ Initialized {len(self.models)} models: {list(self.models.keys())}")

    def get_diverse_models(self, count: int = 3) -> List:
        """Get a diverse set of models for consensus."""
        return random.sample(list(self.models.values()), min(count, len(self.models)))

    def get_best_model(self) -> Any:
        """Get the highest performing model for reflection."""
        for model_name in ['groq_llama3_70b', 'together_llama3', 'ollama_llama3']:
            if model_name in self.models:
                return self.models[model_name]
        return list(self.models.values())[0] if self.models else None

# --- GENERALIZED TOOLS ---

def _generate_search_variants(query: str) -> List[str]:
    """Generate generic search query variations."""
    return [
        query,
        f"facts about {query}",
        f"what is {query}",
        f"detailed explanation of {query}"
    ]

@tool
def enhanced_multi_search(query: str) -> str:
    """Autonomous search with multiple strategies and sources."""
    print("--- 🧠 Searching for information... ---")
    all_results = []
    
    # Strategy 1: Web search with generic variations
    if os.getenv("TAVILY_API_KEY"):
        search_variants = _generate_search_variants(query)
        for variant in search_variants[:2]:  # Limit to 2 variants for speed
            try:
                search_tool = TavilySearchResults(max_results=3)
                docs = search_tool.invoke({"query": variant})
                for doc in docs:
                    all_results.append(f"<WebResult url='{doc.get('url', '')}'>{doc.get('content', '')}</WebResult>")
            except Exception:
                continue
    
    # Strategy 2: Wikipedia search
    try:
        docs = WikipediaLoader(query=query, load_max_docs=2).load()
        for doc in docs:
            all_results.append(f"<WikiResult title='{doc.metadata.get('title', '')}'>{doc.page_content}</WikiResult>")
    except Exception:
        pass # Fail silently if Wikipedia search fails

    if not all_results:
        print("--- ⚠️ No information found. ---")
        return "Comprehensive search did not yield any results."
    
    return "\n\n---\n\n".join(all_results)


# --- CORE AGENT LOGIC ---

class ConsensusVotingSystem:
    """Implements a generalized multi-agent consensus voting system."""
    def __init__(self, model_manager: MultiModelManager):
        self.model_manager = model_manager
        self.reflection_agent = self._create_reflection_agent()

    def _create_reflection_agent(self):
        """Create a specialized reflection agent for answer validation."""
        best_model = self.model_manager.get_best_model()
        return {'model': best_model, 'prompt': REFLECTION_SYSTEM_PROMPT} if best_model else None

    async def get_consensus_answer(self, query: str, search_results: str) -> str:
        """Get consensus answer from multiple agents."""
        models = self.model_manager.get_diverse_models()
        if not models:
            return "No models available for consensus."
        
        tasks = [self._query_single_agent(model, query, search_results) for model in models]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        valid_responses = [res for res in responses if isinstance(res, str) and "agent error" not in res.lower()]
        if not valid_responses:
            return "All agents failed to generate a response based on the provided information."
            
        consensus_answer = self._apply_general_consensus_voting(valid_responses)
        
        if self.reflection_agent:
            return await self._validate_with_reflection(consensus_answer, query)
        return consensus_answer

    async def _query_single_agent(self, model, query: str, search_results: str) -> str:
        """Query a single agent with a standard prompt."""
        try:
            prompt = f"Question: {query}\n\nAvailable Information:\n{search_results}\n\nBased ONLY on the information above, provide the exact answer requested."
            response = await model.ainvoke([SystemMessage(content=CONSENSUS_SYSTEM_PROMPT), HumanMessage(content=prompt)])
            answer = response.content.strip()
            return answer.split("FINAL ANSWER:")[-1].strip() if "FINAL ANSWER:" in answer else answer
        except Exception as e:
            return f"Agent error: {e}"

    def _apply_general_consensus_voting(self, responses: List[str]) -> str:
        """Apply a simple majority vote to the cleaned responses."""
        if not responses: return "Unable to determine a consensus."
        cleaned = [r.strip().lower() for r in responses if r]
        if not cleaned: return "No valid responses to form a consensus."
        return Counter(cleaned).most_common(1)[0][0].capitalize()

    async def _validate_with_reflection(self, answer: str, query: str) -> str:
        """Validate answer using the general-purpose reflection agent."""
        if not self.reflection_agent: return answer
        try:
            prompt = f"Original Question: {query}\nProposed Answer: {answer}\n\nValidate this answer."
            response = await self.reflection_agent['model'].ainvoke([SystemMessage(content=self.reflection_agent['prompt']), HumanMessage(content=prompt)])
            result = response.content.strip()
            if "CORRECTED:" in result: return result.split("CORRECTED:")[-1].strip()
            if "VALIDATED:" in result: return result.split("VALIDATED:")[-1].strip()
            return answer
        except Exception:
            return answer

# --- GRAPH DEFINITION AND EXECUTION ---

class AgentState(TypedDict):
    query: str
    search_results: str
    final_answer: str

class AutonomousLangGraphSystem:
    """Fully autonomous system with a simple search -> consensus graph."""
    def __init__(self):
        self.model_manager = MultiModelManager()
        self.consensus_system = ConsensusVotingSystem(self.model_manager)
        self.graph = self._build_graph()
        print("🚀 Autonomous Multi-Agent System Initialized")

    def _build_graph(self) -> StateGraph:
        g = StateGraph(AgentState)
        g.add_node("search", self._search_node)
        g.add_node("consensus", self._consensus_node)
        g.set_entry_point("search")
        g.add_edge("search", "consensus")
        g.add_edge("consensus", END)
        return g.compile()

    def _search_node(self, state: AgentState) -> dict:
        search_results = enhanced_multi_search.invoke({"query": state["query"]})
        return {"search_results": search_results}

    async def _consensus_node(self, state: AgentState) -> dict:
        print("--- 🤝 Starting consensus panel... ---")
        consensus_answer = await self.consensus_system.get_consensus_answer(
            state['query'], state['search_results']
        )
        return {"final_answer": consensus_answer}
    
    def process_query(self, query: str) -> str:
        """Public-facing method to process a query by running the graph."""
        initial_state = {"query": query}
        config = {"configurable": {"thread_id": f"agent_{time.time()}"}}
        try:
            # We must use asyncio.run to execute the async consensus node
            final_state = asyncio.run(self.graph.ainvoke(initial_state, config))
            return final_state.get("final_answer", "An unknown error occurred during processing.")
        except Exception as e:
            print(f"A critical error occurred: {e}")
            return "Failed to process the request due to a system error."

if __name__ == "__main__":
    system = AutonomousLangGraphSystem()
    
    while True:
        question = input("\nEnter your question (or type 'exit' to quit): ")
        if question.lower() in ['exit','quit','bye']  :
            break
        print("\n⏳ Processing...")
        answer = system.process_query(question)
        print("\n" + "="*50)
        print(f"✅ Final Answer: {answer}")
        print("="*50)
