FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip && \
    pip install --no-cache-dir .

COPY runs ./runs

EXPOSE 8501

CMD ["streamlit", "run", "src/webapp/app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
