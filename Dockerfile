FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /brever

COPY requirements.txt ./requirements.txt
COPY setup.py ./setup.py
COPY brever/ ./brever/
COPY scripts/ ./scripts/
COPY config/ ./config/

RUN pip install -r requirements.txt --no-cache-dir
