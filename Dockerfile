FROM python:3.11-slim

LABEL org.opencontainers.image.title="OpsArena OpenEnv"
LABEL org.opencontainers.image.description="Cloud Ops Command Centre — OpenEnv benchmark"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Create results directory
RUN mkdir -p results

# Set environment variables (defaults)
ENV AGENT_NAME=greedy
ENV API_BASE_URL=""
ENV MODEL_NAME="gpt-4-turbo"
ENV HF_TOKEN=""

# Expose port (default for HF Spaces)
EXPOSE 7860

# Launch FastAPI app with mounted Gradio
CMD ["uvicorn", "app.interface:app", "--host", "0.0.0.0", "--port", "7860"]