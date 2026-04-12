# Occupancy API – Docker image for Azure (App Service, Container Apps, ACI)
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update
RUN apt-get install -y --no-install-recommends curl gnupg ca-certificates

# From https://learn.microsoft.com/en-us/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server?view=sql-server-ver15&tabs=debian18-install%2Calpine17-install%2Cdebian8-install%2Credhat7-13-install%2Crhel7-offline#debian18
# Download the package to configure the Microsoft repo
RUN curl -sSL -O https://packages.microsoft.com/config/debian/$(grep VERSION_ID /etc/os-release | cut -d '"' -f 2 | cut -d '.' -f 1)/packages-microsoft-prod.deb && \
  # Install the package
  dpkg -i packages-microsoft-prod.deb && \
# Delete the file
  rm packages-microsoft-prod.deb && \
  apt-get update && \
  ACCEPT_EULA=Y apt-get install -y msodbcsql18 mssql-tools18 libgssapi-krb5-2 && \ 
  echo 'export PATH="$PATH:/opt/mssql-tools18/bin"' >> ~/.bashrc

RUN pip install --no-cache-dir -r requirements.txt

# Application
COPY main.py .
COPY ml/ ml/

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
