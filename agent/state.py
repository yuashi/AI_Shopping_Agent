from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    query_type: str  # set by the router: product_search | image_search | product_qa | policy_qa | order_status | chitchat