# AdLint web app. Local: docker build -t adlint . && docker run -p 8080:8080 adlint
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY adlint ./adlint
COPY webapp ./webapp
COPY models ./models
COPY lambda_handler.py .

RUN useradd --create-home adlint
USER adlint
ENV ADLINT_WORK_DIR=/tmp/adlint PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn webapp.app:app --host 0.0.0.0 --port ${PORT}"]
