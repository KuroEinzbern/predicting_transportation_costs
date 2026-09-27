FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ARG MODEL_VERSION
ENV MODEL_VERSION=${MODEL_VERSION}

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*


RUN mkdir -p models && \
    curl -L "https://huggingface.co/KuroEinzbern/transportation_cost/resolve/main/model_${MODEL_VERSION}?download=true" -o models/model_${MODEL_VERSION}


COPY . .
RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "challenge_spotter.api.api_endpoints:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
