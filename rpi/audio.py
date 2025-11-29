import argparse
import os
import sys
import threading
import numpy as np
import cv2
import pyaudio
import time
from stage2_inference_and_performance import camera

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- Shared Events ----
camera_event = threading.Event()

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
        while True:
            current_rms = get_audio_rms(stream)
            if(current_rms > AUDIO_THRESHOLD):
                if(time.time() - audio_start_time < TOGGLE_COOLDOWN): #  Avoid rapid toggling
                    print("NOT YET")
                    continue
                elif not camera_event.is_set():
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
def main(args):
    t_worker = threading.Thread(target=camera, args=(camera_event, args),daemon=True)
    t_audio = threading.Thread(target=audio_listener, daemon=True)

    t_worker.start()
    t_audio.start()

    t_worker.join()
    t_audio.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stage 2 Adaptive Controller Inference')

    # Data
    parser.add_argument('--data_dir', type=str, default='../data/clean',
                        help='Path to the directory containing test data.')

    # Model/Checkpoint
    parser.add_argument('--checkpoint', type=str, 
                        default='best_controller_12layers.pth',
                        help='Path to the trained Stage 2 controller checkpoint.')
    parser.add_argument('--total_layers', type=int, default=12,
                        help='Total layer budget used during Stage 2 training.')

    args = parser.parse_args()

    main(args)