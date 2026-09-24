# Base image with Python 3.12 (matches the README / CI test matrix)
FROM python:3.12-slim

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Runtime libraries only: all Python dependencies ship as binary wheels,
# so no compiler toolchain is needed. ffmpeg is used for HLS/RTSP/YouTube decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY app/ ./app/

# Run as an unprivileged user
RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /app/app/data && chown -R appuser:appuser /app/app/data
USER appuser

# Expose port
EXPOSE 8000

# Liveness probe (/healthz is exempt from Basic auth)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"

# Run the FastAPI server via Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
