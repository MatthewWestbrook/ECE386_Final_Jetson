import sys
import time
import requests
import subprocess
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Pipeline, pipeline
import torch
import numpy as np
import numpy.typing as npt
import sounddevice as sd
import Jetson.GPIO as GPIO
import ollama



LLM_MODEL: str = "gemma3:27b"  # Optional, change this to be the model you want
OLLAMA_HOST = "http://ai.dfec.xyz:11434"  # Ollama server URL
# Initialize the client
client = ollama.Client(host=OLLAMA_HOST)


LLM_prompt = """
## Instructions
I need to extract a location based on a question regarding the weather at that location.
The location can be one of three options:
1. A city
2. An airport
3. a generic location that is NOT a city

### Notes
If you ever need to use a space, just use a '+' instead
If the location is an airport, return the location as the three letter IATA designation of the airport
If the location is generic and not a city, place a '~' before the answer

### Examples
'what is the weather at the Eiffel Tower?' will lead to response '~Eiffel+Tower' (option 3)
'I would like to know how warm it is in Colorado Springs' will lead to response 'Colorado+Springs' (option 1)
'I would like to know if it is going to rain at Denver International Airport' which will lead to response 'dia' (option 2)
'What is the weather at Ball Arena?' which will lead to response '~Ball+Arena' (option 3)
'I want to know the weather at The Albuquerque International Sunport' which will lead to response 'abq' (option 2)
'How cold will it be in Seattle?' which will lead to response 'Seattle' (option 1)  

### Additional Information
You will first check to see if the location you extract is a city, if it is not, you will check to see if it is an airport.
If that is not true, you will then extract just the generic location.
Reminder that it can only be one of the three cases!
A response should never end in a '+'.
Do not say anything else, just say the response I am asking for.

### Prompt
The prompt you will extract the location from is:
"""

def record_audio(duration_seconds: int = 10) -> npt.NDArray:
    """Record duration_seconds of audio from default microphone.
    Return a single channel numpy array."""
    sample_rate = 16000  # Hz
    samples = int(duration_seconds * sample_rate)
    audio = sd.rec(samples, samplerate=sample_rate, channels=1, dtype=np.float32)
    sd.wait()
    return np.squeeze(audio, axis=1)
    
def build_pipeline(
    model_id: str, torch_dtype: torch.dtype, device: str
) -> Pipeline:
    """Creates a Hugging Face automatic-speech-recognition pipeline on the given device."""
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe

def llm_parse_for_wttr(prompt: str) -> str:
    """Uses the Ollama client to parse the prompt and extract the weather location."""
    response = client.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    
    return response['message'].content.strip()

def curl_result(Vectorized_Text: str):
    raw_input = LLM_prompt + Vectorized_Text

    try:
        result = llm_parse_for_wttr(raw_input).strip()
    except Exception as e:
        print("💥 ERROR:", e)
        return
    print("About to curl the result: ",result)
    command = ['curl', f'https://wttr.in/{result}']
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)

def on_button_pressed():
    print("🔴 Button pressed! Starting recording...")
    audio = record_audio()
    print("Done Recording")

    print("Transcribing...")
    start_time = time.time_ns()
    speech = pipe(audio)
    end_time = time.time_ns()
    print("Done Transcribing")
    
    curl_result(speech["text"])

    print("Waiting for button press. Press Ctrl+C to exit.")

if __name__ == "__main__":

    model_id = "distil-whisper/distil-medium.en"
    print(f"Using model_id {model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using device {device}.")

    print("Building model pipeline...")
    pipe = build_pipeline(model_id, torch_dtype, device)
    print(type(pipe))
    print("Done")

    BUTTON_PIN = 29  # BOARD numbering

    # Setup GPIO
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING, callback=lambda pin: on_button_pressed(), bouncetime=1000)

    try:
        print("Waiting for button press. Press Ctrl+C to exit.")
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Exiting...")
        GPIO.cleanup()
