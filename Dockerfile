FROM ubuntu:22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml ./

RUN uv python install 3.11 && uv sync --no-dev

COPY . ./

ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["uv", "run", "-m", "blax_ml_mcp.main"]