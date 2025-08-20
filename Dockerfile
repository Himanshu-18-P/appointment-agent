FROM python:3.9-slim

WORKDIR /app

# ----------------------------
# 1. Install system dependencies (for sentence-transformers/faiss)
# ----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# 2. Install torch (CPU version) first to prevent CUDA bloat
# ----------------------------
RUN pip install --upgrade pip \
 && pip install torch==2.1.2+cpu --index-url https://download.pytorch.org/whl/cpu

# ----------------------------
# 3. Copy requirements and install Python deps
# ----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------
# 4. Copy application code and create necessary folders
# ----------------------------
COPY . .
RUN mkdir -p /app/bots_data /app/vector_store

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
