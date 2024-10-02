# Dockerfile for Streamlit app
FROM python:3.12-slim

# Wprking dir
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install streamlit requests streamlit-drawable-canvas Pillow

# Copy the app code
COPY code/deployment/app/app.py /app/app.py

EXPOSE 8501

# Run app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
