FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Alapvető adatfeldolgozó csomagok és JupyterLab telepítése egy lépésben
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    jupyterlab

RUN pip install --no-cache-dir \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu126

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]