# import Jetson.GPIO as GPIO
# import time

# BUTTON_PIN = 29

# GPIO.setmode(GPIO.BOARD)
# GPIO.setup(BUTTON_PIN, GPIO.IN)


# try:
#     while True:
#         state = GPIO.input(BUTTON_PIN)
#         print("Waiting for press..." if state else "Pressed!")
#         time.sleep(0.5)
# except KeyboardInterrupt:
#     GPIO.cleanup()

import Jetson.GPIO as GPIO
import subprocess
import time

BUTTON_PIN = 29  # BOARD numbering

GPIO.setmode(GPIO.BOARD)
GPIO.setup(BUTTON_PIN, GPIO.IN)

def on_button_pressed():
    print("Button pressed! Triggering bash command...")
    subprocess.run(["bash", "/home/ai/Desktop/ECE386_Final/crush.sh"])

GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING, callback=lambda pin: on_button_pressed(), bouncetime=300)

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    GPIO.cleanup()
