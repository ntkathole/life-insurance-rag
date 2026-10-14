FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY feature_repo/ feature_repo/
COPY app/ app/
COPY data/ data/
COPY scripts/ scripts/

EXPOSE 7860

CMD ["python", "app/demo_app.py"]
