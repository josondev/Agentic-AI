   # Multi-Agent LLM Council with Consensus Voting

This project implements a sophisticated multi-agent Large Language Model (LLM) system that leverages the latest research in agentic workflows to achieve high-performance evaluations. It uses a consensus voting mechanism across a diverse panel of LLM agents, followed by a reflection step to ensure answer accuracy and robustness. The entire system is orchestrated using LangGraph.

## Key Features

*   **Multi-Agent Consensus**: Deploys a panel of diverse LLM agents (from Groq, Ollama, Together AI, etc.) that work in parallel to answer a given query. A specialized voting system then determines the most accurate response.
*   **Reflection & Validation**: A dedicated reflection agent examines the consensus answer to verify its logical consistency, factual accuracy, and format, correcting it if necessary.
*   **Enhanced Multi-Source Search**: A custom search tool gathers context by performing multiple actions in parallel:
    *   Utilizes pre-loaded domain knowledge for known question patterns.
    *   Performs web searches with multiple query variations via Tavily Search.
    *   Queries Wikipedia for encyclopedic information.
*   **Diverse & Open-Source Model Support**: Easily integrates with multiple LLM providers, prioritizing fast, open-source models.
    *   **Groq**: Llama3 (70B & 8B), Mixtral
    *   **Ollama**: Local models like Llama3, Mistral, Qwen2
    *   **Together AI**: Hosted models like Llama-3-70b-chat-hf
*   **Domain-Specific Logic**: Implements enhanced fallback and validation logic tailored to specific question types, dramatically improving accuracy on benchmark tasks.
*   **Stateful Graph Architecture**: Built with LangGraph, providing a robust and clear execution flow for complex agent interactions.

## System Architecture

The system follows a structured, graph-based workflow to process each query:

1.  **Routing**: The initial query is routed to the consensus-based processing pipeline.
2.  **Enhanced Search**: The `enhanced_multi_search` tool is invoked to gather comprehensive information from its various sources.
3.  **Multi-Agent Invocation**: The system selects a diverse set of up to nine models from the `MultiModelManager`. Each model acts as an independent agent, receiving the query and the search results to formulate an answer.
4.  **Consensus Voting**: The `ConsensusVotingSystem` collects all responses. It applies domain-specific cleaning and voting logic (e.g., finding the most common number, the highest value, or a specific username pattern) to determine the best preliminary answer.
5.  **Reflection**: A high-performing "reflection agent" validates the answer from the voting step against the original question and known patterns, providing a final layer of quality control.
6.  **Final Answer**: The validated (and potentially corrected) answer is returned to the user.

## Installation

1.  **Clone the repository:**
    ```
    git clone https://github.com/josondev/Agentic-AI
    cd Agentic-AI
    ```

2.  **Create a virtual environment (recommended):**
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file with the following content:
    ```
      python-dotenv==1.0.0
      langchain==0.1.0
      langchain-core==0.1.10
      langchain-community==0.0.13
      langchain-groq==0.0.1
      langchain-together==0.0.1
      langchain-ollama==0.0.1
      langgraph==0.0.20
      tavily-python==0.3.0
      wikipedia==1.4.0
      nest-asyncio==1.5.8
    ```
    Then, install the packages:
    ```
    pip install -r requirements.txt
    ```

## Configuration

The system uses API keys for various services. These should be stored in a `.env` file in the root directory.

1.  Create a file named `.env`.
2.  Add The API keys to the file. The system is designed to function even if some keys are not provided.

    ```
    # .env file
    GROQ_API_KEY="your-groq-api-key"
    TAVILY_API_KEY="your-tavily-api-key"
    ```



## How to Run

Save the provided code as a Python file (e.g., `main.py`). You can run the script directly from your terminal to see it process a set of test questions.


