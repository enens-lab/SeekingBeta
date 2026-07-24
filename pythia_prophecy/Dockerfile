# pythia_prophecy - Auth/User Management Backend
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (excluding frontend)
COPY api/ ./api/
COPY data/ ./data/
COPY frontend/src/data/ ./frontend_data/
COPY templates/ ./templates/
COPY scripts/ ./scripts/
COPY main.py .

# Create non-root user for security
RUN useradd -m -u 1000 pythia && chown -R pythia:pythia /app
USER pythia

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/healthz', timeout=5)" || exit 1

# Run the server
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8001"]
