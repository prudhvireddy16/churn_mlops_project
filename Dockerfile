# 1. Use an official Python light-weight image
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file into the container
COPY requirements.txt .

# 4. Install the libraries inside the container
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your folders into the container
COPY api/ api/
COPY models/ models/

# 6. Expose the port FastAPI runs on
EXPOSE 8000

# 7. Command to run the API when the container starts
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]