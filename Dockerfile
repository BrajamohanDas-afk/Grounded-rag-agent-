FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[anthropic,openai,ollama,local,ui]"
COPY app ./app
COPY data ./data
COPY eval ./eval
EXPOSE 8000 8501
# default: API. docker-compose runs the UI as a second service.
CMD ["uvicorn", "grounded_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
