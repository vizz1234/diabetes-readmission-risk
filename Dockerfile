# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies for libraries like LightGBM / SHAP
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies into user directory
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Clean runtime environment
FROM python:3.11-slim as runner

WORKDIR /app

# Install runtime OpenMP package needed for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from builder stage
COPY --from=builder /root/.local /root/.local

# Copy application source and placeholders
COPY src/ /app/src/
COPY artifacts/ /app/artifacts/
COPY mlruns/ /app/mlruns/
COPY data/ /app/data/

# Add local bin to path and configure python path
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app/src

EXPOSE 8000

# Run uvicorn server
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
