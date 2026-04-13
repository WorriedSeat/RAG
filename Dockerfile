FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --timeout 300 \
    torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --timeout 600 --retries 5 \
    llama-cpp-python==0.3.16

RUN pip install --no-cache-dir --timeout 300 -r requirements.txt

COPY config ./config
COPY src ./src

# API port
EXPOSE 8000
# Streamlit port
EXPOSE 8501

CMD ["uvicorn", "src.deployment.api.api:app", "--host", "0.0.0.0", "--port", "8000"]
