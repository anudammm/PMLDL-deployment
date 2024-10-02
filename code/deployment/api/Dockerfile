FROM python:3.12-slim

WORKDIR /app

COPY code/deployment/api/requirements.txt .

RUN pip install --upgrade pip

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir -r requirements.txt

COPY code /app/code

RUN mkdir -p /app/models
COPY models/best.pt /app/models/best.pt

EXPOSE 8000

# run API
CMD ["uvicorn", "code.deployment.api.api:app", "--host", "0.0.0.0", "--port", "8000"]



#docker build -t fastapi-digit-recognition -f code/deployment/api/Dockerfile .
#docker run -p 8000:8000 fastapi-digit-recognition