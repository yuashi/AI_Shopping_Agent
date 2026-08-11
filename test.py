import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    params = StdioServerParameters(command=sys.executable, args=["mcp_server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("Discovered tools:", [t.name for t in tools.tools])

            result = await session.call_tool(
                "search_products", {"query": "hydrating face serum", "top_k": 2}
            )
            print("\nsearch_products result:")
            print(result.content[0].text)

            result2 = await session.call_tool(
                "answer_policy_question", {"query": "how fast is shipping"}
            )
            print("\nanswer_policy_question result:")
            print(result2.content[0].text[:300])


if __name__ == "__main__":
    asyncio.run(main())