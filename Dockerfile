# FROM nvcr.io/nvidia/pytorch:25.02-py3-igpu
# WORKDIR /app
# RUN apt update && \
#     apt install -y --no-install-recommends libportaudiocpp0 libportaudio2 && \
#     pip install --upgrade --no-cache-dir pip && \
#     pip install --no-cache-dir \
#     transformers \
#     accelerate \
#     sounddevice
# COPY speech_recognition.py .
# ENV HF_HOME="/huggingface/"
# ENTRYPOINT ["python", "speech_recognition.py"]

FROM nvcr.io/nvidia/pytorch:25.02-py3-igpu

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt update && \
    apt install -y --no-install-recommends \
        portaudio19-dev \
        libportaudiocpp0 \
        libportaudio2 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        libffi-dev \
        libgpiod-dev \
        && apt clean && rm -rf /var/lib/apt/lists/*

# Install Jetson.GPIO manually from NVIDIA repo
RUN pip install --no-cache-dir \
    git+https://github.com/NVIDIA/jetson-gpio.git

# Install required Python packages
RUN pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir \
        transformers==4.49.0 \
        accelerate==1.5.2 \
        sounddevice \
        ollama


# Copy project files
COPY speech_recognition.py .
COPY hot_start.sh .

# Ensure the script is executable
RUN chmod +x hot_start.sh

# HuggingFace cache location
ENV HF_HOME="/huggingface/"

# Run the app
ENTRYPOINT ["python", "speech_recognition.py"]
