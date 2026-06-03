FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# awscli for the S3 artifact bootstrap (download_artifacts.sh). Pip install keeps
# the image slim vs the bundled installer; only needed when AUTO_DOWNLOAD_ARTIFACTS=true.
RUN pip install --no-cache-dir awscli

# Copy application code
COPY . .

# Copy and make download script executable
COPY scripts/download_artifacts.sh /app/scripts/download_artifacts.sh
RUN chmod +x /app/scripts/download_artifacts.sh

# Create artifacts directory
RUN mkdir -p /app/artifacts

# Create non-root user
RUN useradd -m -u 1000 pythia && chown -R pythia:pythia /app

USER pythia

CMD ["sh", "-c", "if [ \"${AUTO_DOWNLOAD_ARTIFACTS:-false}\" = \"true\" ]; then /app/scripts/download_artifacts.sh; fi; exec python main.py --host 0.0.0.0"]
