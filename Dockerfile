FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NFLREADPY_HOST=0.0.0.0 \
    NFLREADPY_PORT=8000

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache-dir .

EXPOSE 8000

CMD ["python", "-m", "nflreadpy.api_server"]
