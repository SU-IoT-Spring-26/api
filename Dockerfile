# Occupancy API – Docker image for Azure (App Service, Container Apps, ACI)
FROM python:3.12-slim

WORKDIR /app

# Install system deps for ODBC/SQL Server (unixODBC, msodbcsql18) and Python requirements
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg ca-certificates unixodbc unixodbc-dev && \
    mkdir -p /etc/apt/keyrings && \
    curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /etc/apt/keyrings/microsoft.gpg && \
    chmod 644 /etc/apt/keyrings/microsoft.gpg && \
    curl https://packages.microsoft.com/config/debian/12/prod.list | sed 's|^deb |deb [signed-by=/etc/apt/keyrings/microsoft.gpg] |' > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt

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
