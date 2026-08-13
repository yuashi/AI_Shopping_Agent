"""
Run standalone for debugging:
    mcp dev mcp_server.py
"""
import builtins
import sys
import json

_real_print = print


def _stderr_print(*args, **kwargs):
    kwargs["file"] = sys.stderr
    _real_print(*args, **kwargs)


builtins.print = _stderr_print

from mcp.server.fastmcp import FastMCP

from agent import tools_core

tools_core._get_product_embedder()
tools_core._get_image_embedder()
tools_core._get_policy_embedder()

mcp = FastMCP("shopping-tools")

from agent import tools_core

mcp = FastMCP("shopping-tools")


@mcp.tool()
def search_products(
    query: str,
    min_rating: float = 0.0,
    max_price: float = -1.0,
    top_k: int = 5,
) -> str:
    """Search products.

    Args:
        query: Product description, e.g. "black backpack" or "chocolate".
        min_rating: Minimum product rating from 0 to 5.
        max_price: Maximum price. Use -1 for no price limit.
        top_k: Maximum number of products to return.
    """

    result = tools_core.search_products(
        query=query,
        min_rating=min_rating,
        max_price=None if max_price < 0 else max_price,
        top_k=top_k,
    )

    return json.dumps(result)


@mcp.tool()
def search_by_image(image_description: str, top_k: int = 5) -> list[dict]:
    """Find products similar to an uploaded image using a text description."""

    result = tools_core.search_by_image_description(
        image_description=image_description,
        top_k=top_k,
    )

    return json.dumps(result)


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
    """Look up the mocked status of an order."""

    result = tools_core.check_order_status(
        order_id=order_id
    )

    return json.dumps(result)


if __name__ == "__main__":
    # mcp.run(transport="stdio")
    print("MCP: running server", file=sys.stderr, flush=True)
    mcp.run(transport="stdio")

