from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

QUERY_TYPES = ["product_search", "image_search", "product_qa", "policy_qa", "order_status", "chitchat"]

SYSTEM_PROMPTS = {
    "product_search": (
        "You are a helpful shopping assistant. Use the search_products tool to find "
        "matching items. Present results as a short list with title, price, and rating, "
        "and briefly explain why each one fits the request."
    ),
    "image_search": (
        "You are a helpful shopping assistant. The user uploaded an image. Use the "
        "search_by_image tool with a description of the image to find visually similar "
        "products, then present the best matches with a brief note on why they're similar."
    ),
    "product_qa": (
        "You are a helpful shopping assistant answering a question about a specific "
        "product. Use get_product_reviews_summary and/or search_products to ground your "
        "answer in real data. Be honest about mixed or negative feedback if present."
    ),
    "policy_qa": (
        "You are a helpful shopping assistant answering a shipping/returns/warranty/ "
        "payment question. Use the answer_policy_question tool and answer only from what "
        "it returns — do not invent policy details."
    ),
    "order_status": (
        "You are a helpful shopping assistant. Use the check_order_status tool to look up "
        "the order and report its status clearly and concisely."
    ),
    "chitchat": (
        "You are a friendly shopping assistant for an online store. Respond briefly and "
        "steer the conversation back to how you can help with products, orders, or policies."
    ),
}

QUERY_TYPE_TOOLS = {
    "product_search": ["search_products"],
    "image_search": ["search_by_image"],
    "product_qa": ["get_product_reviews_summary", "search_products"],
    "policy_qa": ["answer_policy_question"],
    "order_status": ["check_order_status"],
    "chitchat": [],
}

class QueryClassification(BaseModel):
    query_type: str = Field(description=f"One of: {', '.join(QUERY_TYPES)}")


CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Classify the user's latest shopping-assistant message into exactly one category: "
     f"{', '.join(QUERY_TYPES)}. Consider the conversation for context (e.g. a follow-up "
     "question about a product already discussed is product_qa)."),
    ("placeholder", "{messages}"),
])


def make_router_node(llm):

    classifier = (
        CLASSIFY_PROMPT
        | llm.with_structured_output(QueryClassification)
    )

    async def router_node(state):

        result = await classifier.ainvoke(
            {
                "messages": state["messages"]
            }
        )

        return {
            "query_type": result.query_type
        }

    return router_node


def get_system_prompt(query_type: str) -> str:
    return SYSTEM_PROMPTS.get(query_type, SYSTEM_PROMPTS["chitchat"])

def get_relevant_tools(query_type: str, all_tools: list) -> list:
    """Filter the full MCP tool list down to just what this query type needs."""
    wanted = QUERY_TYPE_TOOLS.get(query_type)
    if not wanted:  # chitchat, or an unrecognized type — fall back to everything
        return all_tools
    return [t for t in all_tools if t.name in wanted]