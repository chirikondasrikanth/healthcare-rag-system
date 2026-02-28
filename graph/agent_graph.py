import sys
sys.path.append('.')

from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from agents.retrieval_agent import retrieval_agent
from agents.validation_agent import validation_agent
from agents.response_agent import response_agent

# Define state
class AgentState(TypedDict):
    query: str
    retrieved_chunks: List[dict]
    retrieval_done: bool
    validation_passed: bool
    validation_message: str
    final_answer: str
    sources: List[str]

def should_continue(state):
    """Route based on validation"""
    if state.get("validation_passed"):
        return "response_agent"
    else:
        return END

def build_graph():
    """Build multi agent graph"""
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("retrieval_agent", retrieval_agent)
    graph.add_node("validation_agent", validation_agent)
    graph.add_node("response_agent", response_agent)
    
    # Add edges
    graph.set_entry_point("retrieval_agent")
    graph.add_edge("retrieval_agent", "validation_agent")
    graph.add_conditional_edges(
        "validation_agent",
        should_continue,
        {
            "response_agent": "response_agent",
            END: END
        }
    )
    graph.add_edge("response_agent", END)
    
    return graph.compile()

def run_agent(query):
    """Run the multi agent pipeline"""
    print(f"\n{'='*50}")
    print(f"Query: {query}")
    print(f"{'='*50}")
    
    graph = build_graph()
    
    result = graph.invoke({
        "query": query,
        "retrieved_chunks": [],
        "retrieval_done": False,
        "validation_passed": False,
        "validation_message": "",
        "final_answer": "",
        "sources": []
    })
    
    print(f"\n📋 Final Answer: {result['final_answer']}")
    print(f"📚 Sources: {result['sources']}")
    return result

if __name__ == "__main__":
    run_agent("Who is eligible for Medicare?")
    run_agent("What does Medicare Part A cover?")
    run_agent("What is medigap?")