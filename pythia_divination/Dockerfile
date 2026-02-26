FROM python:3.12-slim

WORKDIR /app

# Install system dependencies including AWS CLI
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    groff \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI
RUN pip install --no-cache-dir awscli

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy and make download script executable
COPY scripts/download_artifacts.sh /app/scripts/download_artifacts.sh
RUN chmod +x /app/scripts/download_artifacts.sh

# Create artifacts directory
RUN mkdir -p /app/artifacts

# Download artifacts from S3 at startup
RUN /app/scripts/download_artifacts.sh

# Create non-root user
RUN useradd -m -u 1000 pythia && chown -R pythia:pythia /app

USER pythia

CMD ["python", "main.py", "--host", "0.0.0.0"]
