import operator
from typing import TypedDict, List, Annotated, Union, Optional
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):    
    messages: Annotated[List[Union[BaseMessage, dict]], operator.add]
    intent: str
    context: str
    pending_amendment: Union[str, None]
    approval: Optional[bool]