# Imagen reproducible (Python 3.11)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Solo modo paper: no hay claves, no se envían órdenes.
CMD ["python", "-m", "src.run", "--mode", "paper"]
