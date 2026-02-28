def validation_agent(state):
    """Agent 2 - Validates retrieved content"""
    print("✔️ Validation Agent running...")
    
    chunks = state.get("retrieved_chunks", [])
    query = state["query"]
    
    # Check if chunks are relevant
    if not chunks:
        state["validation_passed"] = False
        state["validation_message"] = "No relevant content found"
        return state
    
    # Check minimum chunks
    if len(chunks) < 1:
        state["validation_passed"] = False
        state["validation_message"] = "Insufficient context"
        return state
    
    # Check if query is healthcare related
    healthcare_keywords = [
        "medicare", "medicaid", "health", "medical", "insurance",
        "coverage", "hospital", "doctor", "prescription", "drug",
        "eligib", "enroll", "premium", "deductible", "benefit"
    ]
    
    query_lower = query.lower()
    is_healthcare = any(k in query_lower for k in healthcare_keywords)
    
    if not is_healthcare:
        # Still allow but flag it
        state["validation_passed"] = True
        state["validation_message"] = "General query - answering from context"
    else:
        state["validation_passed"] = True
        state["validation_message"] = "Healthcare query validated"
    
    print(f"✅ Validation: {state['validation_message']}")
    return state