FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for numpy/pandas/scikit-learn
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Hugging Face models so they’re baked into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "from transformers import pipeline; pipeline('text2text-generation', model='google/flan-t5-base')"

# Copy your app
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
