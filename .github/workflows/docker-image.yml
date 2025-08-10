FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends dcm2niix \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
ENTRYPOINT ["python","centiloid_pipeline.py"]
