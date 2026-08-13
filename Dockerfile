FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build data + indexes at image build time so the Space starts up fast
RUN python data/build_amazon_dataset.py && \
    EMBEDDER_BACKEND=sentence_transformers python data/build_product_index.py && \
    EMBEDDER_BACKEND=sentence_transformers python data/build_policy_index.py && \
    EMBEDDER_BACKEND=sentence_transformers python data/build_image_index.py

EXPOSE 8501

ENV EMBEDDER_BACKEND=sentence_transformers

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]