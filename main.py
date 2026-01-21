"""
Multi-Agent AI System - Terminal Version
Updated with current Groq models (January 2025)
"""

import os
import time
import random
import operator
from typing import List, Dict, Any, TypedDict, Annotated
from collections import Counter
import asyncio
import nest_asyncio
from dotenv import load_dotenv

# Apply the patch
nest_asyncio.apply()

from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

load_dotenv()

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

# --- MODEL MANAGEMENT WITH UPDATED MODELS ---

class MultiModelManager:
    """Manages multiple Groq LLM models with current supported models"""
    def __init__(self):
        self.models = {}
        self._initialize_models()

    def _initialize_models(self):
        """Initialize available Groq models - UPDATED FOR JANUARY 2025"""
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not groq_api_key:
            raise RuntimeError("❌ GROQ_API_KEY not found in environment variables")
        
        try:
            # PRODUCTION MODELS - High quality, reliability, and speed
            
            # Llama 3.3 70B Versatile - Most capable production model
            self.models['llama3.3_70b'] = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                api_key=groq_api_key
            )
            
            # Llama 3.1 8B Instant - Fastest production model
            self.models['llama3.1_8b'] = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.1,
                api_key=groq_api_key
            )
            
            # Llama Guard 3 8B - Safety and content moderation
            self.models['gpt-oss'] = ChatGroq(
                model="openai/gpt-oss-20b",
                temperature=0.1,
                api_key=groq_api_key
            )
            
            # PREVIEW MODELS - For evaluation (add diversity)
            
            try:
                # Llama 4 Maverick 17B - Latest Llama 4 preview
                self.models['llama4_maverick'] = ChatGroq(
                    model="meta-llama/llama-4-maverick-17b-128e-instruct",
                    temperature=0.1,
                    api_key=groq_api_key
                )
            except:
                pass
            
            try:
                # Llama 4 Scout 17B - Faster Llama 4 variant
                self.models['llama4_scout'] = ChatGroq(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    temperature=0.1,
                    api_key=groq_api_key
                )
            except:
                pass
            
            try:
                # Qwen 3 32B - Alternative reasoning model
                self.models['qwen3_32b'] = ChatGroq(
                    model="qwen/qwen3-32b",
                    temperature=0.1,
                    api_key=groq_api_key
                )
            except:
                pass
            
            if not self.models:
                raise RuntimeError("❌ No models could be initialized")
            
            print(f"✅ Initialized {len(self.models)} models: {list(self.models.keys())}")
            
        except Exception as e:
            raise RuntimeError(f"❌ Failed to initialize Groq models: {e}")

    def get_diverse_models(self, count: int = 3) -> List:
        available_count = min(count, len(self.models))
        return random.sample(list(self.models.values()), available_count)

    def get_best_model(self) -> Any:
        # Priority order based on production quality and capabilities
        for model_name in ['llama3.3_70b', 'llama4_maverick', 'qwen3_32b', 'llama4_scout', 'llama3.1_8b']:
            if model_name in self.models:
                return self.models[model_name]
        return list(self.models.values())[0] if self.models else None

# --- SEARCH TOOL ---

@tool
def enhanced_multi_search(query: str) -> str:
    """Search with Tavily and Wikipedia"""
    all_results = []
    
    # Tavily Search
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
                
            print(f"✓ Tavily: Found {len(all_results)} results")
        except Exception as e:
            print(f"⚠️  Tavily search error: {e}")
    
    # Wikipedia Search
    try:
        docs = WikipediaLoader(query=query, load_max_docs=1).load()
        for doc in docs:
            content = doc.page_content[:1500]
            title = doc.metadata.get('title', 'Wikipedia')
            all_results.append(f"<WikiResult title='{title}'>{content}</WikiResult>")
        print(f"✓ Wikipedia: Found {len(docs)} result(s)")
    except Exception as e:
        print(f"ℹ️  Wikipedia: {e}")
    
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
            print(f"  → Agent {agent_num}: Generated response ({len(extracted)} chars)")
            return extracted
        except Exception as e:
            print(f"  → Agent {agent_num}: Error - {str(e)[:100]}")
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
            thinking_log.append(f"⚠️  Reflection failed: {str(e)[:50]}")
            return answer

# --- GRAPH SYSTEM ---

class AgentState(TypedDict):
    query: str
    search_results: str
    thinking_log: Annotated[List[str], operator.add]
    final_answer: str

class AutonomousLangGraphSystem:
    def __init__(self):
        print("\n🚀 Initializing Multi-Agent System...")
        self.model_manager = MultiModelManager()
        self.consensus_system = ConsensusVotingSystem(self.model_manager)
        self.graph = self._build_graph()
        print("✅ System Ready!\n")

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
        initial_state = {"query": query, "thinking_log": []}
        config = {"configurable": {"thread_id": f"agent_{time.time()}"}}
        try:
            final_state = asyncio.run(self.graph.ainvoke(initial_state, config))
            return {
                "answer": final_state.get("final_answer", "Processing error"),
                "thinking_log": final_state.get("thinking_log", [])
            }
        except Exception as e:
            return {
                "answer": f"Error: {e}", 
                "thinking_log": [f"Error: {e}"]
            }

# --- MAIN CLI INTERFACE ---

if __name__ == "__main__":
    print("=" * 70)
    print("        🤖 AGENTIC AI SYSTEM - TERMINAL INTERFACE")
    print("=" * 70)
    print("\n  Multi-Agent Consensus System with Web Search")
    print("  Updated with latest Groq models (Jan 2025)")
    print("  Type 'exit', 'quit', or 'q' to stop\n")
    
    try:
        system = AutonomousLangGraphSystem()
        
        while True:
            print("-" * 70)
            query = input("\n💬 Your Question: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!\n")
                break
            
            if not query:
                print("⚠️  Please enter a question")
                continue
            
            print(f"\n⏳ Processing '{query}'...\n")
            
            output = system.process_query(query)
            
            # Display thinking process
            print("\n🧠 THINKING PROCESS:")
            print("─" * 70)
            for entry in output['thinking_log']:
                print(f"   {entry}")
            
            # Display final answer
            print("\n" + "=" * 70)
            print("💡 FINAL ANSWER:")
            print("=" * 70)
            print(f"\n{output['answer']}\n")
            print("=" * 70)
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}\n")
        import traceback
        traceback.print_exc()
