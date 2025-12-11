def get_latest_user_input(messages):    
    if not messages:
        return ""
    
    last_msg = messages[-1]
    content = (
        last_msg.get("content", "") 
        if isinstance(last_msg, dict) 
        else getattr(last_msg, "content", "")
    )
    
    if isinstance(content, list):
        return " ".join([
            c["text"] if isinstance(c, dict) and "text" in c else str(c) 
            for c in content
        ])
    
    return str(content)

def route_step(state):
    intent = state.get("intent", "QUERY")
    if intent == "AMEND":
        return "draft_amendment"
    elif intent == "GREETING":
        return "conversation"
    else:
        return "retrieve"