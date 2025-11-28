import os
import sys
import threading

from sklearn import pipeline
from rpi.key_listener import key_listener
from rpi.res import worker
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import random
import numpy as np
import pyrealsense2 as rs
import cv2
import pyaudio
import time
import psutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- Shared Events ----
camera_event = threading.Event()
stop_program_event = threading.Event()

# --- Audio Settings ---
AUDIO_THRESHOLD = 800
TOGGLE_COOLDOWN = 5.0
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# ---- Audio RMS Calculation ----
def get_audio_rms(stream):
    try:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float64)
        rms = np.sqrt(np.mean(audio_data**2))  # RMS calculation (volume)
        return rms
    except Exception as e:
        return 0

# ---- Audio Listener Thread ----
def audio_listener():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT, 
        channels=CHANNELS, 
        rate=RATE, 
        input=True,             
        frames_per_buffer=CHUNK
    )
    audio_start_time = time.time()
    try:
        while not stop_program_event.is_set():
            current_rms = get_audio_rms(stream)
            if(time.time() - audio_start_time < TOGGLE_COOLDOWN): #  Avoid rapid toggling
                continue
            elif(current_rms > AUDIO_THRESHOLD):
                if not camera_event.is_set():
                    print("Audio trigger detected: START camera")
                    camera_event.set()
                    audio_start_time = time.time()
                else:
                    print("Audio trigger detected: STOP camera")
                    camera_event.clear()
                    audio_start_time = time.time()
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# ---- Main ----
def main():
    t_worker = threading.Thread(target=worker, args=(camera_event, stop_program_event),daemon=True)
    t_audio = threading.Thread(target=audio_listener, daemon=True)

    t_worker.start()
    t_audio.start()

    t_worker.join()
    t_audio.join()

if __name__ == "__main__":
    main()