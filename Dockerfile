FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    INPUT_DIR=/data/input \
    OUTPUT_DIR=/data/output \
    HF_HOME=/cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils libcairo2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir --no-build-isolation causal-conv1d flash-linear-attention

COPY extract.py watcher.py excel_export.py ./

RUN mkdir -p /data/input /data/output /cache/huggingface

VOLUME ["/data/input", "/data/output", "/cache/huggingface"]

CMD ["python", "watcher.py"]
