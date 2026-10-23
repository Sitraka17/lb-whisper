FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git build-essential cmake && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# requirements.txt carries the CPU wheel indexes for torch and llama-cpp-python;
# build-essential and cmake are a fallback if no llama-cpp-python wheel matches.
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV GRADIO_SERVER_NAME=0.0.0.0 PORT=7860 LB_MODEL_CACHE=/models
VOLUME ["/models"]
EXPOSE 7860
CMD ["python", "app.py"]
