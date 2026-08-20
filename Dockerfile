FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for matplotlib
# fonts-noto-cjk：讓 matplotlib 畫圖時中文字不會變成缺字方塊
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libcurl4-openssl-dev \
    libssl-dev \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install lightgbm openpyxl

# Copy source code
COPY *.py .

# Create data directory for watchlist persistence
RUN mkdir -p /app/data

CMD ["python", "bot.py"]
