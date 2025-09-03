FROM python:slim

ENV PYTHONDONTWRITEBBYTECODE = 1 \
    PYTHONUNBUFFERED = 1

WORKDIR /app

RUN apt-get update && apt-get-install -y --no-install-recommends \
    libgomp1 \
    && apt-get-clean \
    && rm -rf /var/lib/apt/lists/*


COPY . .

RUN pip install --no-cashe-dir -e .

RUN python pipeline/training_pipeline.python

EXPOSE 5000

CMD ["python", "application.py"]