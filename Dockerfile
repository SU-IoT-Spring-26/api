# Occupancy API – Docker image for Azure (App Service, Container Apps, ACI)
FROM python:3.12-slim

WORKDIR /app

# Install dependencies (no build tools needed for current deps)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application
COPY main.py .

# Default data directory inside container (can override with THERMAL_DATA_DIR)
ENV THERMAL_DATA_DIR=/app/thermal_data
# Azure App Service sets PORT; default for local runs
ENV PORT=8000

# Create data directory so the app can write without root
RUN mkdir -p /app/thermal_data

EXPOSE 8000

# Optional: health check for Azure/orchestrators (GET /api/test)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=2 \
  CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/api/test')"]

# Run with python main.py so PORT from env is used (Azure sets PORT)
CMD ["python", "main.py"]
