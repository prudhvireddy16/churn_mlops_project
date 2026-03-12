FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
# Run uvicorn from the root of /app
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]