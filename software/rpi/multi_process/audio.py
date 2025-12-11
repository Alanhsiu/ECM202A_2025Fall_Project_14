import argparse
import os
import queue
import sys
import threading
import numpy as np
import pyaudio
import time
import multiprocessing as mp
from multiprocessing import shared_memory
import torch

from ui import ui_generate
from camera import camera
from stage2_inference import inference_init



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- Process Shared Events ----

stop_event    = mp.Event()   
camera_ready = mp.Event()
camera_event  = mp.Event()
model_ready  = mp.Event()

inference_start = mp.Event()
inference_ended = mp.Event()

result_queue = mp.Queue()
log_queue = mp.Queue() 

num_slots = 2
SHAPE = (480, 640, 3)
DTYPE = np.uint8



# --- Audio Settings ---
AUDIO_THRESHOLD = 3500
TOGGLE_COOLDOWN = 5.0
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# clear flash

def flush_stream(stream, duration=0.2):
    frames = int(duration * RATE / CHUNK)
    for _ in range(frames):
        try:
            stream.read(CHUNK, exception_on_overflow=False)
        except:
            pass


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
    os.sched_setaffinity(0, {0})
    p = pyaudio.PyAudio()
    camera_ready.wait()
    model_ready.wait()
    stream = p.open(
        format=FORMAT, 
        channels=CHANNELS, 
        rate=RATE, 
        input=True,             
        frames_per_buffer=CHUNK
    )


    audio_start_time = time.time()-TOGGLE_COOLDOWN
    try:
        while True:
            flush_stream(stream, duration=0.05)
            if stop_event.is_set():
                break
            current_rms = get_audio_rms(stream)
            if(current_rms > AUDIO_THRESHOLD):
                if(time.time() - audio_start_time < TOGGLE_COOLDOWN): #  Avoid rapid toggling
                    log_queue.put("[Audio] NOT YET")
                    continue
                elif not camera_event.is_set():
                    log_queue.put("[Audio] Audio trigger detected: START camera")
                    camera_event.set()
                    audio_start_time = time.time()
                else:
                    log_queue.put("[Audio] Audio trigger detected: STOP camera")    
                    camera_event.clear()
                    audio_start_time = time.time()
        time.sleep(0.05)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def ui_display(shm_name, shape, dtype_str, num_slots, inference_start, inference_ended, result_queue, log_queue):
    os.sched_setaffinity(0, {0})
    low_light_event = threading.Event()
    shared_lock   = threading.Lock()
    shared_state  = {
        "frame": None,      # camera thread write / UI thread read (np.ndarray, shape 480x1280x3)
        "layer_rgb": None,
        "layer_depth":None,
        "last_result": None # dict: {"pred": ..., "rgb": ..., "depth": ..., "latency": ...}
    }

    t_camera = threading.Thread(target=camera, args=(shm_name, shape, dtype_str, num_slots,inference_start, inference_ended, result_queue, stop_event, camera_event, low_light_event, camera_ready, shared_lock, shared_state, log_queue), daemon=True)
    t_ui = threading.Thread(target=ui_generate, args=(camera_event, stop_event, low_light_event, shared_state, shared_lock, log_queue), daemon=True)
    
    t_camera.start()
    t_ui.start()
    
    t_camera.join()
    t_ui.join()
    

# ---- Main ----
def main(args):
    # mp.set_start_method("spawn", force=True)

    nbytes = num_slots * np.prod(SHAPE) * np.dtype(DTYPE).itemsize
    shm = shared_memory.SharedMemory(create=True, size=nbytes)

    big_arr = np.ndarray((num_slots, *SHAPE), dtype=DTYPE, buffer=shm.buf) # (2*3*224*224)

    rgb_layer = big_arr[0]   
    depth_layer = big_arr[1]

    rgb_layer[:] = 0
    depth_layer[:] = 0
    dtype_str = np.dtype(DTYPE).name
    shm_name = shm.name

    p_inference = mp.Process(target=inference_init, args=(shm_name, SHAPE, dtype_str, num_slots, result_queue, inference_start, inference_ended, model_ready, stop_event, log_queue, args), daemon=True)
    p_audio = mp.Process(target=audio_listener, daemon=True)
    p_ui = mp.Process(target=ui_display, args=(shm_name, SHAPE, dtype_str, num_slots,inference_start,inference_ended,result_queue, log_queue), daemon=True)
    
    p_inference.start()
    p_audio.start()
    p_ui.start()


    p_audio.join()
    p_ui.join()
    p_inference.join()
    shm.close()
    shm.unlink()


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
