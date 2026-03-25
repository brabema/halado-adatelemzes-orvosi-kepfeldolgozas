# Setup

## Előfeltételek

- Python 3.11+
- Docker Desktop
- Kaggle account (adatletöltéshez)

## Adatok letöltése

### Automatikus (Kaggle API)

1. Hozz létre API tokent: Kaggle → Settings → Create New Token
2. Másold a letöltött `kaggle.json` fájlt ide: `~/.kaggle/kaggle.json`
3. Futtasd:

```bash
pip install kaggle
python scripts/download_data.py
```

### Manuális

1. Töltsd le az adatokat innen: https://www.kaggle.com/datasets/benedekbrandschott/halado-adatelemzes-orvosi-kepfeldolgozas-dataset
2. Helyezd a zip fájlokat a `datas/` mappába

## Docker

### Indítás

```bash
docker compose up --build
```

### Kaggle hozzáférés konténerben

Ha a konténerből szeretnéd letölteni az adatokat, vedd ki a kommentet a `docker-compose.yaml`-ból:

```yaml
- ~/.kaggle:/root/.kaggle
```
