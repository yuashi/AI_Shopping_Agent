import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.router import get_relevant_tools,get_system_prompt, make_router_node
from agent.state import AgentState

load_dotenv()

MCP_SERVER_PATH = str(
    Path(__file__).parent.parent / "mcp_server.py"
)


def get_llm():
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "1024")),
    )


async def build_graph():

    from langchain_mcp_adapters.client import (
        MultiServerMCPClient
    )

    print("Creating MCP client...", flush=True)

    client = MultiServerMCPClient(
        {
            "shopping": {
                "command": sys.executable,
                "args": [MCP_SERVER_PATH],
                "transport": "stdio",
            }
        }
    )

    print("Getting MCP tools...", flush=True)

    tools = await client.get_tools()

    print(
        f"MCP tools loaded: {[tool.name for tool in tools]}",
        flush=True
    )

    llm = get_llm()
    router_node = make_router_node(llm)

    print("Creating LLM...", flush=True)

    # llm_with_tools = llm.bind_tools(tools)

    # router_node = make_router_node(llm)

    async def agent_node(state: AgentState):

        query_type = state.get("query_type", "chitchat")
        system_prompt = get_system_prompt(query_type)
        relevant_tools = get_relevant_tools(query_type, tools)
        llm_with_tools = llm.bind_tools(relevant_tools) if relevant_tools else llm
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("router")

    workflow.add_edge(
        "router",
        "agent"
    )

    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        }
    )

    workflow.add_edge(
        "tools",
        "agent"
    )

    checkpointer = MemorySaver()

    compiled_graph = workflow.compile(
        checkpointer=checkpointer
    )

    print("Graph compiled", flush=True)

    return compiled_graph