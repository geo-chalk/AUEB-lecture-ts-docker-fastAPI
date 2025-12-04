# setup/Dockerfile
FROM python:3.10-slim

# Install system libs (LightGBM needs libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy dependency files first (Caching layer)
COPY pyproject.toml uv.lock README.md ./

# Install dependencies
# --frozen: Error if lockfile is out of sync
# --no-dev: Exclude dev dependencies (pytest, black, etc.)
# --no-install-project: Don't install the current project yet, just libs
RUN uv sync --frozen --no-dev --no-install-project

# Enable the virtual environment
# uv creates a .venv folder. We just add it to the path.
ENV PATH="/app/.venv/bin:$PATH"

# Copy source code
COPY src/ ./src/

# Security: Non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

#ENTRYPOINT ["uv", "run", "--frozen", "--no-dev", "./src/main.py"]

# =====================
# --- FastAPI part
# =====================
# Expose the port the app runs on
EXPOSE 8000

# Run the application
# host 0.0.0.0 is CRITICAL for Docker networking
CMD ["uvicorn", "--app-dir", "src", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]