FROM python:3.8-slim

WORKDIR /deploy

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY config.json .
COPY app.py .

EXPOSE 5000

CMD ["python", "app.py", "5000"]
