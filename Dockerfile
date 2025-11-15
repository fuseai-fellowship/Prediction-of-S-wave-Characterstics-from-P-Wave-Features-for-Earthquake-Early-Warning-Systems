# Small Dockerfile to run the Streamlit app
FROM python:3.10-slim

WORKDIR /app

# system deps (if obspy needs them, keep minimal here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

EXPOSE 8501

# default start (overridable)
CMD ["bash", "scripts/run_app.sh"]