"""
MCP server exposing the shopping agent's tools over stdio.

Run standalone for debugging:
    mcp dev mcp_server.py
Or it gets launched automatically by agent/graph.py via MultiServerMCPClient.
"""
from mcp.server.fastmcp import FastMCP

from agent import tools_core

mcp = FastMCP("shopping-tools")


@mcp.tool()
def search_products(query: str, min_rating: float = 0.0, max_price: float = -1, top_k: int = 5) -> list[dict]:
    """Search the product catalog by natural-language description. Use min_rating
    (0-5) and max_price (-1 means no limit) to filter results."""
    return tools_core.search_products(
        query=query,
        min_rating=min_rating,
        max_price=None if max_price < 0 else max_price,
        top_k=top_k,
    )


@mcp.tool()
def search_by_image(image_description: str, top_k: int = 5) -> list[dict]:
    """Find products similar to an uploaded image, given a short description of
    what's in the image. Note: for the real pixel-based CLIP nearest-neighbor
    search, the Streamlit app calls agent.tools_core.search_by_image directly
    with the actual image, bypassing the LLM — this tool is the LLM-facing
    text-description fallback for when the agent needs to reason about an
    image conversationally without the raw bytes."""
    return tools_core.search_by_image_description(image_description=image_description, top_k=top_k)


@mcp.tool()
def get_product_reviews_summary(product_id: str) -> str:
    """Summarize customer reviews for a given product_id."""
    return tools_core.get_product_reviews_summary(product_id=product_id)


@mcp.tool()
def answer_policy_question(query: str) -> str:
    """Retrieve relevant shipping / returns / warranty / payment policy text for
    a customer's question."""
    return tools_core.answer_policy_question(query=query)


@mcp.tool()
def check_order_status(order_id: str) -> dict:
    """Look up the (mocked) status of an order by order_id."""
    return tools_core.check_order_status(order_id=order_id)


if __name__ == "__main__":
    mcp.run(transport="stdio")