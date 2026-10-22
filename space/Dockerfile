FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY . .
ENV GRADIO_SERVER_NAME=0.0.0.0 PORT=7860 LB_MODEL_CACHE=/models
VOLUME ["/models"]
EXPOSE 7860
CMD ["python", "app.py"]
