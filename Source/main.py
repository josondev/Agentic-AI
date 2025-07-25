"""
Ultra-Enhanced Multi-Agent LLM System with Consensus Voting
Implements latest 2024-2025 research for maximum evaluation performance.
This version is fully autonomous and includes a transparent thinking process.
"""

import os
import time
import random
import operator
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

CONSENSUS_SYSTEM_PROMPT = """You are part of a multi-agent expert panel. Your role is to provide the most accurate and precise answer possible based ONLY on the provided information.
    EXTRACTION RULES:
- Parse all relevant data, names, and numbers from the search results.
- Cross-reference information from multiple sources to verify facts.
- Use contextual reasoning to resolve ambiguities.
- If the information is insufficient, state that clearly.

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

@tool
def enhanced_multi_search(query: str) -> str:
    """Autonomous search with multiple strategies and sources."""
    all_results = []
    if os.getenv("TAVILY_API_KEY"):
        search_variants = [query, f"facts about {query}"]
        for variant in search_variants:
            try:
                search_tool = TavilySearchResults(max_results=2)
                docs = search_tool.invoke({"query": variant})
                for doc in docs:
                    all_results.append(f"<WebResult url='{doc.get('url', '')}'>{doc.get('content', '')}</WebResult>")
            except Exception: continue
    try:
        docs = WikipediaLoader(query=query, load_max_docs=1).load()
        for doc in docs:
            all_results.append(f"<WikiResult title='{doc.metadata.get('title', '')}'>{doc.page_content}</WikiResult>")
    except Exception: pass
    if not all_results: return "Comprehensive search did not yield any results."
    return "\n\n---\n\n".join(all_results)


# --- CORE AGENT LOGIC ---

class ConsensusVotingSystem:
    """Implements a generalized multi-agent consensus voting system."""
    def __init__(self, model_manager: MultiModelManager):
        self.model_manager = model_manager
        self.reflection_agent = self._create_reflection_agent()

    def _create_reflection_agent(self):
        best_model = self.model_manager.get_best_model()
        return {'model': best_model, 'prompt': REFLECTION_SYSTEM_PROMPT} if best_model else None

    async def get_consensus_answer(self, query: str, search_results: str, thinking_log: List[str]) -> str:
        models = self.model_manager.get_diverse_models()
        if not models:
            thinking_log.append("❌ No models available for consensus.")
            return "No models available for consensus."
        
        thinking_log.append(f"🤝 Starting consensus round with {len(models)} agents...")
        tasks = [self._query_single_agent(model, query, search_results) for model in models]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        valid_responses = [res for res in responses if isinstance(res, str) and "agent error" not in res.lower()]
        thinking_log.append(f"📝 Raw answers from agents: {valid_responses}")

        if not valid_responses:
            thinking_log.append("❌ All agents failed to generate a response.")
            return "All agents failed to generate a response based on the provided information."
            
        consensus_answer = self._apply_general_consensus_voting(valid_responses, thinking_log)
        
        if self.reflection_agent:
            return await self._validate_with_reflection(consensus_answer, query, thinking_log)
        return consensus_answer

    async def _query_single_agent(self, model, query: str, search_results: str) -> str:
        try:
            prompt = f"Question: {query}\n\nAvailable Information:\n{search_results}\n\nBased ONLY on the information above, provide the exact answer requested."
            response = await model.ainvoke([SystemMessage(content=CONSENSUS_SYSTEM_PROMPT), HumanMessage(content=prompt)])
            answer = response.content.strip()
            return answer.split("FINAL ANSWER:")[-1].strip() if "FINAL ANSWER:" in answer else answer
        except Exception as e:
            return f"Agent error: {e}"

    def _apply_general_consensus_voting(self, responses: List[str], thinking_log: List[str]) -> str:
        thinking_log.append("🗳️ Applying consensus voting...")
        if not responses: return "Unable to determine a consensus."
        cleaned = [r.strip().lower() for r in responses if r]
        if not cleaned:
            thinking_log.append("⚠️ No valid responses to form a consensus.")
            return "No valid responses to form a consensus."
        
        vote_counts = Counter(cleaned)
        most_common = vote_counts.most_common(1)[0]
        thinking_log.append(f"📊 Vote counts: {vote_counts.most_common()}")
        thinking_log.append(f"✅ Consensus answer is '{most_common[0]}' with {most_common[1]} votes.")
        return most_common[0].capitalize()

    async def _validate_with_reflection(self, answer: str, query: str, thinking_log: List[str]) -> str:
        if not self.reflection_agent: return answer
        thinking_log.append("🤔 Reflecting on the consensus answer...")
        try:
            prompt = f"Original Question: {query}\nProposed Answer: {answer}\n\nValidate this answer."
            response = await self.reflection_agent['model'].ainvoke([SystemMessage(content=self.reflection_agent['prompt']), HumanMessage(content=prompt)])
            result = response.content.strip()
            thinking_log.append(f"🕵️ Reflection agent output: {result}")
            if "CORRECTED:" in result: return result.split("CORRECTED:")[-1].strip()
            if "VALIDATED:" in result: return result.split("VALIDATED:")[-1].strip()
            return answer
        except Exception as e:
            thinking_log.append(f"⚠️ Reflection agent failed: {e}")
            return answer

# --- GRAPH DEFINITION AND EXECUTION ---

class AgentState(TypedDict):
    query: str
    search_results: str
    thinking_log: Annotated[List[str], operator.add]
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
        log = ["🧠 Searching for information..."]
        search_results = enhanced_multi_search.invoke({"query": state["query"]})
        log.append("✅ Search complete.")
        return {"search_results": search_results, "thinking_log": log}

    async def _consensus_node(self, state: AgentState) -> dict:
        log = []
        consensus_answer = await self.consensus_system.get_consensus_answer(
            state['query'], state['search_results'], log
        )
        return {"final_answer": consensus_answer, "thinking_log": log}
    
    def process_query(self, query: str) -> Dict:
        """Public-facing method to process a query and return answer with log."""
        initial_state = {"query": query, "thinking_log": []}
        config = {"configurable": {"thread_id": f"agent_{time.time()}"}}
        try:
            # Use asyncio.run to execute the async graph and get the final state.
            # ainvoke runs the full graph and returns the final state, including the accumulated log.
            final_state = asyncio.run(self.graph.ainvoke(initial_state, config))
            return {
                "answer": final_state.get("final_answer", "Processing resulted in an error."),
                "thinking_log": final_state.get("thinking_log", [])
            }
        except Exception as e:
            return {"answer": f"A critical error occurred: {e}", "thinking_log": [f"Error: {e}"]}


if __name__ == "__main__":
    system = AutonomousLangGraphSystem()
    
    while True:
        question = input("\nEnter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        print("\n⏳ Processing...")
        output = system.process_query(question)
        print("\n" + "="*50)
        print("🤔 Thinking Process Log:")
        for entry in output['thinking_log']:
            print(f"   {entry}")
        print("\n" + "="*50)
        print(f"✅ Final Answer: {output['answer']}")
        print("="*50)
