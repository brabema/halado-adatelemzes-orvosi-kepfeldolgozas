FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libxcb1 \
        libx11-6 \
        libxext6 \
        libsm6 \
        libxrender1 \
        libgl1 \
        libglib2.0-0 \
        libgl1-mesa-glx \
        libxkbcommon0 \
        libxcb-render0 \
        libxcb-shm0 \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt --no-cache-dir

ENV PYTHONPATH=/app/app:/app

WORKDIR /app/app
CMD ["python", "main.py"]
