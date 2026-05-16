# Multi-stage build for production-ready volatility API

# Stage 1: Builder
FROM python:3.12-slim as builder

WORKDIR /tmp

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ src/
COPY setup.py .

# Create non-root user for security
RUN useradd -m -u 1000 volatility && \
    chown -R volatility:volatility /app

USER volatility

# Set PATH to include pip --user packages
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO \
    RESULTS_DIR=/app/results

# Create results directory
RUN mkdir -p /app/results

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run service
CMD ["uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8000"]
