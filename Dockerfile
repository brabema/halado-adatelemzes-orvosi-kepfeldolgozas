FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

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

# Alapvető adatfeldolgozó csomagok és JupyterLab telepítése egy lépésben
RUN pip install -r requirements.txt --no-cache-dir

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]