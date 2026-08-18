# ShopAssist — Multi-Modal AI Shopping Agent

An Amazon-style shopping assistant built with **LangGraph**, **LangChain**, **RAG**, and **MCP**.
It searches products (specifically Electronics) by natural-language description, rating, and price; finds visually
similar products from an uploaded image; answers product questions grounded in reviews;
and answers shipping/returns/warranty/payment questions grounded in policy documents —
entirely on a free stack.

## Deployed Version

[ShopAssist]https://ai-shoppingagent.streamlit.app/

## Dataset

McAuley-Lab/Amazon-Reviews-2023 from Hugging Face
(Category - Electronics)

## Models Used

- Langraph Agent -> openai/gpt-oss-120b
- Transformer Text Embedder -> all-MiniLM-L6-v2
- CLIP Image Embedder -> ViT-B-32

## Architecture

```
Streamlit UI (chat + image upload)
        │
        ▼
LangGraph agent (router → tool-calling agent loop, with conversation memory)
        │
        ▼
MCP server (mcp_server.py) exposing 5 tools over stdio:
  - search_products          (Chroma vector search over product catalog)
  - search_by_image          (CLIP-based visual similarity search)
  - get_product_reviews_summary
  - answer_policy_question   (Chroma vector search over policy docs)
  - check_order_status       (mocked transactional tool)
        │
        ▼
ChromaDB (persisted locally) — two collections: `products`, `policies`
```

## Setup

```bash
git clone https://github.com/yuashi/AI_Shopping_Agent.git
cd AI_Shopping_Agent
pip install -r requirements.txt

# 1. Generate the dataset
python data/generate_synthetic_data.py
# ... or the real Amazon Reviews 2023 dataset (needs internet + `pip install "datasets==3.6.0"`)
python data/build_amazon_dataset.py --category Electronics --n_products 5000

# 2. Build the vector indexes
python data/build_product_index.py
python data/build_policy_index.py
python data/build_image_index.py

# You can also skip the data generation and indexing (step 1. and 2.) competely and use the existing chroma_db embeddings instead.

# 3. Get a free Groq API key: https://console.groq.com
cp .env.example .env
# edit .env and add your GROQ_API_KEY

# 4. Run it
streamlit run app.py
```

## Repo structure

```
shopping-agent/
├── data/
│   ├── generate_synthetic_data.py   # synthetic dataset
│   ├── build_amazon_dataset.py      # real Amazon Reviews 2023 loader
│   ├── build_product_index.py
│   ├── build_policy_index.py
│   ├── build_image_index.py         # CLIP (or mock) product-photo index
│   └── policies/*.md
│   └── products.json                # Products
│   └── reviews.json                 # Reviews
├── agent/
│   ├── embeddings.py     # TF-IDF/mock vs sentence-transformers/CLIP backends
│   ├── tools_core.py     # retrieval + business logic, independently testable
│   ├── state.py          # LangGraph state schema
│   ├── router.py         # query classifier + system prompts + per-turn tool filtering
│   ├── graph.py          # the LangGraph StateGraph itself
│   └── startup.py        # auto-builds/checks data/indexes on first run (for Streamlit Cloud)
├── chroma_db/            # chroma db with the embeddings
├── mcp_server.py          # MCP server wrapping agent/tools_core.py
├── app.py                 # Streamlit UI
├── requirements.txt
├── Dockerfile              # for hosts that still support free Docker
└── .env.example
```

## Author

Ayushi Singh
