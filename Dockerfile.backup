FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY sql/ ./sql/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi[standard]>=0.123.0 \
    uvicorn>=0.30.0 \
    psycopg2-binary>=2.9.9 \
    python-dotenv>=1.0.0 \
    pandas>=2.2.0 \
    numpy>=1.26.0 \
    requests>=2.31.0 \
    google-genai>=1.0.0 \
    openpyxl>=3.1.0 \
    pydantic[email]>=2.0.0 \
    tabulate>=0.9.0 \
    matplotlib>=3.8.0 \
    seaborn>=0.13.0 \
    scikit-learn>=1.4.0

# Expose port
EXPOSE 8000

# Run the application - Railway will set PORT env var
CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}
