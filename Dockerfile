FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive \
    API_URL=http://localhost:8000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (UID 1000 is required by Hugging Face)
RUN useradd -m -u 1000 user
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/ ./src/
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY data/ ./data/
COPY models/ ./models/
COPY params.yaml .
COPY mlruns/ ./mlruns/
COPY start.sh .

# Adjust permissions so user 1000 can read, write, and execute
RUN chmod +x start.sh && chown -R user:user /app

# Switch to the non-root user
USER user

# Expose port 8501 (Streamlit default)
EXPOSE 8501

# Run the startup script
CMD ["./start.sh"]
