import asyncio
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from PIL import Image

from agent.graph import build_graph
from agent.tools_core import search_by_image

import logging

logging.getLogger("streamlit").setLevel(logging.ERROR)

st.set_page_config(page_title="ShopAssist — AI Shopping Agent", page_icon="🛍️", layout="wide")


def get_event_loop():
    """One loop for the whole app lifetime — reused for BOTH building the graph
    and every subsequent invocation. The MCP client's stdio subprocess connection
    is bound to whichever loop was live when it was created; running it from a
    different loop later causes the coroutine to hang forever with no error."""
    global _APP_LOOP
    if _APP_LOOP is None:
        _APP_LOOP = asyncio.new_event_loop()
    return _APP_LOOP


_APP_LOOP = None


@st.cache_resource
def load_graph():
    loop = get_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(build_graph())


def run_async(coro):
    loop = get_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []

st.title("🛍️ ShopAssist")
st.caption("Search Electronics, get recommendations, upload a photo, or ask about shipping & returns.")

with st.sidebar:
    st.subheader("Search by image")
    uploaded = st.file_uploader("Upload a product photo", type=["png", "jpg", "jpeg"])
    if uploaded:
        st.image(uploaded, width=150)
    if uploaded and st.button("Find similar products"):
        pil_image = Image.open(uploaded)
        results = search_by_image(pil_image, top_k=5)
        st.session_state.history.append(("user", f"[uploaded image: {uploaded.name}]"))
        if results:
            lines = [f"**{r['title']}** — ⭐ {r['rating']} — ${r['price']}" for r in results]
            st.session_state.history.append(("assistant", "Here are visually similar products:\n\n" + "\n\n".join(lines)))
        else:
            st.session_state.history.append((
                "assistant",
                "No image index found yet — run `python data/build_image_index.py` first.",
            ))
        st.rerun()

    st.divider()
    if st.button("Reset conversation"):
        st.session_state.history = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

if prompt := st.chat_input("Ask about a product, an order, or a policy..."):
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                graph = load_graph()

                config = {
                    "configurable": {
                        "thread_id": st.session_state.thread_id
                    }
                }

                result = run_async(
                    graph.ainvoke(
                        {
                            "messages": [HumanMessage(content=prompt)],
                            "query_type": ""
                        },
                        config=config,
                    )
                )

                final_message = result["messages"][-1]

                answer = (
                    final_message.content
                    if isinstance(final_message, AIMessage)
                    else str(final_message)
                )

        except Exception as e:
            print("Exception while invoking graph:", flush=True)
            print(f"Error details: {e}", flush=True)
            answer = (
                "Sorry, something went wrong while processing your request. "
                "Please try again."
            )

            st.session_state.thread_id = str(uuid.uuid4())

        st.markdown(answer)
    st.session_state.history.append(("assistant", answer))

