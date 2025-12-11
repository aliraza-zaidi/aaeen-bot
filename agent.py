from langchain.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langchain_core.documents import Document
from dotenv import load_dotenv
from state import AgentState
from config import Config
from utility import get_latest_user_input, route_step
from rag import initialize_vector_store, vectorize
from llm import initialize_llm

load_dotenv()
vector_store = initialize_vector_store()
vectorize(vector_store)
retriever = vector_store.as_retriever(search_kwargs={"k": Config.RETRIEVAL_K})
llm = initialize_llm()

def router_node(state: AgentState) -> dict:
    """Classify user intent: QUERY, AMEND, or GREETING."""
    user_input = get_latest_user_input(state["messages"])
    
    system_prompt = (
        "Classify the user's intent:\n"
        "1. 'QUERY' - Constitution questions.\n"
        "2. 'AMEND' - Update/Change/Amend the Constitution.\n"
        "3. 'GREETING' - Hello/Hi/Thanks.\n"
        "Return ONLY the classification word."
    )
    
    classification = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]).content.strip().upper()
    
    # Default to QUERY if classification is invalid
    if classification not in ["QUERY", "AMEND", "GREETING"]:
        classification = "QUERY"
    
    return {"intent": classification}

def general_conversation_node(state: AgentState) -> dict:
    """Handle general greetings and casual conversation."""
    user_input = get_latest_user_input(state["messages"])
    response = llm.invoke(f"Reply politely to this greeting: {user_input}")
    return {"messages": [response]}

def retrieve_node(state: AgentState) -> dict:
    """Retrieve relevant context from the constitution."""
    user_input = get_latest_user_input(state["messages"])
    docs = retriever.invoke(user_input)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context_text}

def generate_answer_node(state: AgentState) -> dict:
    """Generate an answer based on retrieved context."""
    context = state.get("context", "")
    user_input = get_latest_user_input(state["messages"])
    
    prompt = (
        "Based on the Constitution of Pakistan context provided below, "
        "answer the question accurately and concisely.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_input}"
    )
    
    response = llm.invoke(prompt)
    return {"messages": [response]}

def draft_amendment_node(state: AgentState) -> dict:
    """Draft a formal amendment based on user request."""
    user_input = get_latest_user_input(state["messages"])
    
    prompt = (
        f"Draft formal legal text to amend the constitution for: '{user_input}'. "
        "Return ONLY the legal amendment text in proper format."
    )
    draft_text = llm.invoke(prompt).content
    
    msg = (
        f"I have drafted this amendment:\n\n{draft_text}\n\n"
        "Should I save this to the Constitution?"
    )
    
    return {
        "messages": [AIMessage(content=msg)],
        "pending_amendment": draft_text
    }

def approval_node(state: AgentState) -> dict:
    """Request human approval for the amendment (interrupt point)."""
    decision = interrupt({
        "question": "Approve this amendment?",
        "draft": state["pending_amendment"]
    })
    
    # Convert decision to boolean approval
    is_approved = decision in [True, "approve", "yes"]
    return {"approval": is_approved}


def apply_update_node(state: AgentState) -> dict:
    """Apply or reject the amendment based on approval."""
    approval_decision = state.get("approval", False)
    
    if approval_decision:
        new_text = state["pending_amendment"]
        vector_store.add_documents([
            Document(
                page_content=new_text,
                metadata={"source": "user_amendment"}
            )
        ])
        message = "✅ Amendment approved and saved to the constitution."
    else:
        message = "❌ Amendment rejected. No changes were made."
    
    return {"messages": [AIMessage(content=message)]}


workflow = StateGraph(AgentState)
    
# Add all nodes
workflow.add_node("router", router_node)
workflow.add_node("conversation", general_conversation_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_answer_node)
workflow.add_node("draft_amendment", draft_amendment_node)
workflow.add_node("approval", approval_node)
workflow.add_node("apply_update", apply_update_node)
    
# Set entry point
workflow.set_entry_point("router")
    
# Add conditional routing from router
workflow.add_conditional_edges(
        "router",
        route_step,
        {
            "draft_amendment": "draft_amendment",
            "conversation": "conversation",
            "retrieve": "retrieve"
        }
    )
    
    
workflow.add_edge("conversation", END)
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
    

workflow.add_edge("draft_amendment", "approval")
workflow.add_edge("approval", "apply_update")
workflow.add_edge("apply_update", END)
    
graph = workflow.compile()
