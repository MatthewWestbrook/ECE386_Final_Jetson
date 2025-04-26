#!/bin/bash

# Source your bash profile to get full environment (especially needed if Docker is only available there)
# source ~/.bashrc

# docker buildx build . -t whisper
# Now run Docker normally
docker run -it --rm --device=/dev/snd --runtime=nvidia --ipc=host -v huggingface:/huggingface/ whisper



