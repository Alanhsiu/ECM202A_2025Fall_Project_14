import argparse
import os
import sys
import threading
import numpy as np
import cv2
import pyaudio
import time
from stage2_inference_and_performance import camera
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import psutil
from collections import deque
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- Shared Events ----
camera_event = threading.Event()
camera_ready = threading.Event()
# --- Audio Settings ---
AUDIO_THRESHOLD = 3500
TOGGLE_COOLDOWN = 5.0
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100




class cpu_usage(object):
    def __init__(self):
        self.max_points = 60  # last 60 seconds info
        self.update_interval = 1000  

        # deque: auto pop old data when maxlen is reached
        self.cpu_data = deque([psutil.cpu_percent(interval=None)] * self.max_points, maxlen=self.max_points)

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title('CPU Usage Monitor')

        # X axis: 0 to max_points-1
        self.x_axis = list(range(self.max_points))
        self.ax.set_xlim(0, self.max_points - 1) 
        self.ax.set_xticklabels([])
        self.ax.set_xlabel('Past 1 Minute -> Now')

        # Y axis: 0% to 100%
        self.ax.set_ylim(0, 100)
        self.ax.set_ylabel('CPU Usage (%)')

        # Initialize the line/area fill and text label
        self.line, = self.ax.plot(self.x_axis, self.cpu_data, color='tab:blue', linewidth=2)
        self.fill = self.ax.fill_between(self.x_axis, self.cpu_data, color='tab:blue', alpha=0.3)
        self.text_label = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, fontsize=12, fontweight='bold', color='tab:blue')

    def update(self, frame):
        cpu_percent = psutil.cpu_percent(interval=None) # non-blocking, get instant value

        self.cpu_data.append(cpu_percent)
        self.line.set_ydata(self.cpu_data)
        
        self.fill.remove() 
        self.fill = self.ax.fill_between(self.x_axis, self.cpu_data, color='tab:blue', alpha=0.3)
        
        self.text_label.set_text(f"Current: {cpu_percent}%")
        
        return self.line, self.text_label
    
    def start(self):
        try:
            self.ani = animation.FuncAnimation(
                self.fig, 
                self.update,
                interval=self.update_interval, 
                cache_frame_data=False
            )
        
            plt.tight_layout()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.show()
        except KeyboardInterrupt:
            plt.close('all')
            sys.exit(0)

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
    print('Camera is ready')
    audio_start_time = time.time()-TOGGLE_COOLDOWN
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
    t_worker = threading.Thread(target=camera, args=(camera_event,camera_ready, args),daemon=True)
    t_audio = threading.Thread(target=audio_listener, daemon=True)
    
    
    t_worker.start()
    t_audio.start()
    
    cpu = cpu_usage()
    cpu.start()

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
