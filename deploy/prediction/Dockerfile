FROM python:3.8-slim

WORKDIR /deploy

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

COPY dbpedia_model.pth.tar .

COPY vocab.pickle .

COPY dpcnn.py .

COPY model.py .

COPY service.py .

EXPOSE 5001

CMD ["python", "app.py", "5001"]
