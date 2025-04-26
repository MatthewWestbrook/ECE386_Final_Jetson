#!/bin/bash

echo "🔧 Building Docker image..."
docker build -t whisper .

echo "🚀 Running container..."
docker run -it --rm \
  --device /dev/gpiochip0 \
  --device /dev/snd \
  --runtime nvidia \
  --ipc host \
  -v /sys:/sys \
  -v /dev:/dev \
  -v huggingface:/huggingface/ \
  -v "$(pwd)":/app \
  --workdir /app \
  --privileged \
  whisper python3 speech_recognition.py


