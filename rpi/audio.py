import argparse
import os
import queue
import sys
import threading
import numpy as np
import cv2
import pyaudio
import time
from ui import ui_generate
from stage2_inference_and_performance import camera
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import psutil
from collections import deque
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- Shared Events ----
low_light_event = threading.Event()
camera_event = threading.Event()
camera_ready = threading.Event()
stop_event    = threading.Event()   


shared_lock   = threading.Lock()
shared_state  = {
    "frame": None,      # camera thread write / UI thread read (np.ndarray, shape 480x1280x3)
    "layer_rgb": None,
    "layer_depth":None,
    "last_result": None # dict: {"pred": ..., "rgb": ..., "depth": ..., "latency": ..., "cpu": ...}
}

log_queue = queue.Queue() 

# --- Audio Settings ---
AUDIO_THRESHOLD = 3500
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
    camera_ready.wait()
    log_queue.put('Camera is ready')
    audio_start_time = time.time()-TOGGLE_COOLDOWN
    try:
        while True:
            if stop_event.is_set():
                break
            current_rms = get_audio_rms(stream)
            if(current_rms > AUDIO_THRESHOLD):
                if(time.time() - audio_start_time < TOGGLE_COOLDOWN): #  Avoid rapid toggling
                    log_queue.put("NOT YET")
                    continue
                elif not camera_event.is_set():
                    log_queue.put("Audio trigger detected: START camera")
                    camera_event.set()
                    audio_start_time = time.time()
                else:
                    log_queue.put("Audio trigger detected: STOP camera")    
                    camera_event.clear()
                    audio_start_time = time.time()
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# ---- Main ----
def main(args):
    t_camera = threading.Thread(target=camera, args=(stop_event, camera_event, low_light_event, camera_ready, shared_lock, shared_state, log_queue, args), daemon=True)
    t_audio = threading.Thread(target=audio_listener, daemon=True)
    t_ui = threading.Thread(target=ui_generate, args=(camera_event, stop_event, low_light_event, shared_state, shared_lock, log_queue), daemon=True)
    
    t_camera.start()
    t_audio.start()
    t_ui.start()

    # cpu = cpu_usage()
    # cpu.start()

    t_camera.join()
    t_audio.join()
    t_ui.join()
    

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
