import os
import sys

from sklearn import pipeline
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
from models.adaptive_controller import AdaptiveGestureClassifier
from data.gesture_dataset import single_image_transform
from data.gesture_dataset import transform_from_camera

class_names = ['standing', 'left_hand', 'right_hand', 'both_hands']

# --- Audio Settings ---
AUDIO_THRESHOLD = 800
TOGGLE_COOLDOWN = 3.0
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# --- Helper Classes & Functions ---

class LayerPlotter:
    def __init__(self, max_history=50, height=200, width=640, total_layers=12):
        self.max_history = max_history
        self.height = height
        self.width = width
        self.total_layers = total_layers
        self.rgb_history = [0] * max_history
        self.depth_history = [0] * max_history
        self.bg_color = (20, 20, 20)
        self.rgb_color = (0, 0, 255)
        self.depth_color = (255, 200, 0)
        self.grid_color = (50, 50, 50)

        self.total_perc  = 100
        self.cpu_history = [0]*max_history
        self.cpu_color = (0, 255, 0)


    def update(self, rgb_val, depth_val):
        self.rgb_history.pop(0)
        self.rgb_history.append(rgb_val)
        self.depth_history.pop(0)
        self.depth_history.append(depth_val)

    def update_cpu(self, cpu_val):
        self.cpu_history.pop(0)
        self.cpu_history.append(cpu_val)

    def draw(self):
        canvas = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
        step_y = self.height // self.total_layers
        for i in range(0, self.total_layers + 1, 2):
            y = self.height - int((i / self.total_layers) * self.height)
            cv2.line(canvas, (0, y), (self.width, y), self.grid_color, 1)

        step_x = self.width / (self.max_history - 1)

        def draw_poly(history, color, max_val):
            points = []
            for i, val in enumerate(history):
                x = int(i * step_x)
                y = self.height - int((val / max_val) * (self.height - 20)) - 10
                points.append((x, y))
            for i in range(len(points) - 1):
                cv2.line(canvas, points[i], points[i+1], color, 2)
            last_pt = points[-1]
            cv2.putText(canvas, str(int(history[-1])), (last_pt[0]-20, last_pt[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        draw_poly(self.rgb_history, self.rgb_color, self.total_layers)
        draw_poly(self.depth_history, self.depth_color, self.total_layers)
        draw_poly(self.cpu_history, self.cpu_color, 100)
        cv2.putText(canvas, "RGB Layers", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.rgb_color, 1)
        cv2.putText(canvas, "Depth Layers", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.depth_color, 1)
        cv2.putText(canvas, "CPU (%)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.cpu_color, 1)
        return canvas

def conceptual_calculate_flops(total_layers, rgb_layers_used, depth_layers_used):
    BASE_FLOPS = 10.0e9    # Fixed cost (GFLOPs)
    PER_LAYER_FLOPS = 0.5e9 # Cost per ViT layer (GFLOPs)

    total_flops = BASE_FLOPS + \
                  (rgb_layers_used * PER_LAYER_FLOPS) + \
                  (depth_layers_used * PER_LAYER_FLOPS)

    return total_flops / 1e9

def inference_stage2(model, data, device, temperature=0.5):
    model.eval() 
    with torch.no_grad():
        rgb = data['rgb'].to(device)
        depth = data['depth'].to(device)
        
        start_time = time.time()

        logits, layer_allocation = model(
            rgb, depth, temperature=temperature, return_allocation=True
        )

        if device.type == 'cuda':
            torch.cuda.synchronize()

        end_time = time.time()

        _, predicted = torch.max(logits, 1)
        pred_label = class_names[predicted.item()]
        rgb_layers = layer_allocation[0, 0, :].sum().item()
        depth_layers = layer_allocation[0, 1, :].sum().item()

    latency_ms = (end_time - start_time) * 1000
    return pred_label, rgb_layers, depth_layers, latency_ms

# --- Main System Controller ---

class SmartSystemController:
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args
        
        self.state = "SLEEP" 
        self.last_toggle_time = 0
        
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                                  input=True, frames_per_buffer=CHUNK)
        
        self.pipeline = None
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        self.plotter = LayerPlotter(max_history=30, height=200, width=640, total_layers=args.total_layers)

    def get_audio_rms(self):
        try:
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float64)
            rms = np.sqrt(np.mean(audio_data**2))
            return rms
        except Exception as e:
            return 0

    def start_camera(self):
        if self.pipeline is None:
            print(">>> [System] Waking up... Starting Camera...")
            self.pipeline = rs.pipeline()
            self.pipeline.start(self.config)
            time.sleep(1.0)
            print(">>> [System] ACTIVE MODE.")

    def stop_camera(self):
        if self.pipeline is not None:
            print(">>> [System] Going to Sleep... Stopping Camera.")
            self.pipeline.stop()
            self.pipeline = None
            cv2.destroyAllWindows()

    def run(self):
        print("\n" + "="*50)
        print(f"System Started via Audio Control.")
        print(f"Make noise > {AUDIO_THRESHOLD} to toggle SLEEP/ACTIVE.")
        # print("Press 'ESC' in Active window to quit.")
        print("="*50 + "\n")

        try:
            while True:
                # 1. Audio Check
                audio_start_time = time.time()
                current_rms = self.get_audio_rms()
                audio_end_time = time.time()
                audio_latency_ms = (audio_end_time - audio_start_time)*1000
                print(f"Audio latency:{audio_latency_ms:5.1f}ms")
                current_time = time.time()

                if current_rms > AUDIO_THRESHOLD and (current_time - self.last_toggle_time > TOGGLE_COOLDOWN):
                    self.last_toggle_time = current_time
                    print(f"\n!!! Sound Trigger ({current_rms:.1f}) !!!")
                    
                    if self.state == "SLEEP":
                        self.state = "ACTIVE"
                        self.start_camera()
                    else:
                        self.state = "SLEEP"
                        self.stop_camera()
                    
                    self.stream.read(CHUNK, exception_on_overflow=False)

                # 2. State Handling
                if self.state == "ACTIVE":
                    try:
                        frames = self.pipeline.wait_for_frames(timeout_ms=2000)
                        depth_frame = frames.get_depth_frame()
                        color_frame = frames.get_color_frame()

                        if not depth_frame or not color_frame:
                            continue

                        depth_image = np.asanyarray(depth_frame.get_data())
                        color_image = np.asanyarray(color_frame.get_data())
                        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                        data = transform_from_camera(color_image, depth_colormap)
                        pred_label, rgb_layers, depth_layers, latency_ms = inference_stage2(self.model, data, self.device)

                        flops = conceptual_calculate_flops(self.args.total_layers, rgb_layers, depth_layers)

                        self.plotter.update(rgb_layers, depth_layers)
                        self.plotter.update_cpu(psutil.cpu_percent(interval=1))
                        chart_img = self.plotter.draw()

                        combined_camera = np.hstack((color_image, depth_colormap))
                        chart_img_resized = cv2.resize(chart_img, (combined_camera.shape[1], 200))
                        final_display = np.vstack((combined_camera, chart_img_resized))

                        text = f"Pred: {pred_label} | RGB: {int(rgb_layers)} | Depth: {int(depth_layers)} | {latency_ms:.1f} ms"
                        
                        overlay = final_display.copy()
                        cv2.rectangle(overlay, (0, 0), (1280, 60), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.6, final_display, 0.4, 0, final_display)
                        
                        color_text = (0, 255, 0) if pred_label != 'standing' else (255, 255, 255)
                        cv2.putText(final_display, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_text, 2)

                        cv2.imshow('Smart ADMN System', final_display)

                        if cv2.waitKey(1) & 0xFF == 27:
                            break
                    
                    except RuntimeError:
                        continue 

                elif self.state == "SLEEP":
                    sleep_img = np.zeros((200, 600, 3), dtype=np.uint8)
                    cv2.putText(sleep_img, f"SLEEP MODE (Vol: {current_rms:.0f})", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Smart ADMN System', sleep_img)
                    
                    if cv2.waitKey(100) & 0xFF == 27:
                        break

        finally:
            self.stop_camera()
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            cv2.destroyAllWindows()
            print("System Shutdown.")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nLoading Model...")
    model = AdaptiveGestureClassifier(
        num_classes=4,
        adapter_hidden_dim=256,
        total_layers=args.total_layers,
        qoi_dim=128,
        stage1_checkpoint=None
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    system = SmartSystemController(model, device, args)
    system.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Smart ADMN Audio-Triggered System')
    parser.add_argument('--checkpoint', type=str, default='best_controller_12layers.pth')
    parser.add_argument('--total_layers', type=int, default=12)
    args = parser.parse_args()

    main(args)
