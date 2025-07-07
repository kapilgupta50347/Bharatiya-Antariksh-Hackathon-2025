FROM python:3.12-slim

WORKDIR /pm-conc-map

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "app.py"]
