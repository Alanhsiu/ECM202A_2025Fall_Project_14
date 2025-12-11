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
from time import time

# import library to control GPIO
from gpiozero import LED
# ------------------------------

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.adaptive_controller import AdaptiveGestureClassifier
from gesture_dataset import single_image_transform
from gesture_dataset import transform_from_camera

class_names = ['standing', 'left_hand', 'right_hand', 'both_hands']

# Initialize GPIO LED Configuration ---
try:
    led_1 = LED(27)   # red LED GPIO 27
    led_2 = LED(22)   # green LED GPIO 22
    led_3 = LED(17)   # blue LED GPIO 17
    led_4 = LED(5)    # yellow LED GPIO 5
    print("[System] GPIO LEDs initialized.")
except Exception as e:
    print(f"[System] GPIO Init Failed: {e} (Ignore if not on RPi)")
    led_1 = None
    led_2 = None
    led_3 = None
    led_4 = None
# -------------------------------------

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

    def update(self, rgb_val, depth_val):

        self.rgb_history.pop(0)
        self.rgb_history.append(rgb_val)
        self.depth_history.pop(0)
        self.depth_history.append(depth_val)

    def draw(self):

        canvas = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)

        step_y = self.height // self.total_layers
        for i in range(0, self.total_layers + 1, 2):
            y = self.height - int((i / self.total_layers) * self.height)
            cv2.line(canvas, (0, y), (self.width, y), self.grid_color, 1)

        step_x = self.width / (self.max_history - 1)

        def draw_poly(history, color):
            points = []
            for i, val in enumerate(history):
                x = int(i * step_x)
                y = self.height - int((val / self.total_layers) * (self.height - 20)) - 10
                points.append((x, y))
            
            for i in range(len(points) - 1):
                cv2.line(canvas, points[i], points[i+1], color, 2)
                
            last_pt = points[-1]
            cv2.putText(canvas, str(int(history[-1])), (last_pt[0]-20, last_pt[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        draw_poly(self.rgb_history, self.rgb_color)
        draw_poly(self.depth_history, self.depth_color)

        cv2.putText(canvas, "RGB Layers", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.rgb_color, 1)
        cv2.putText(canvas, "Depth Layers", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.depth_color, 1)

        return canvas

# --- Conceptual FLOPs Calculation Function ---
def conceptual_calculate_flops(total_layers, rgb_layers_used, depth_layers_used):

    BASE_FLOPS = 10.0e9    # Fixed cost (GFLOPs)
    PER_LAYER_FLOPS = 0.5e9 # Cost per ViT layer (GFLOPs)

    total_flops = BASE_FLOPS + \
                  (rgb_layers_used * PER_LAYER_FLOPS) + \
                  (depth_layers_used * PER_LAYER_FLOPS)

    return total_flops / 1e9

# Update LED state by new predicted gesture ---
def update_led(label):
    if led_1 is None or led_2 is None or led_3 is None or led_4 is None:
            return
    if label != 'standing':
        led_1.off()
    if label != 'left_hand':
        led_2.off()
    if label != 'right_hand':
        led_3.off()
    if label != 'both_hands':
        led_4.off()

    if label == 'standing':
        if not led_1.is_lit:
            led_1.blink(on_time=0.5, off_time=0, n=1, background=True)
    elif label == 'left_hand':
        if not led_2.is_lit:
            led_2.blink(on_time=0.5, off_time=0, n=1, background=True)
    elif label == 'right_hand':
        if not led_3.is_lit:
            led_3.blink(on_time=0.5, off_time=0, n=1, background=True)
    elif label == 'both_hands':
        if not led_4.is_lit:
            led_4.blink(on_time=0.5, off_time=0, n=1, background=True)
# ---------------------------------------------

def inference_stage2(model, data, device, temperature=0.5):

    model.eval() 
    with torch.no_grad():
        rgb = data['rgb'].to(device)
        depth = data['depth'].to(device)

        # Core: Measure model runtime (Latency)
        start_time = time()

        logits, layer_allocation = model(
            rgb, 
            depth, 
            temperature=temperature,
            return_allocation=True
        )

        if device.type == 'cuda':
            torch.cuda.synchronize()

        end_time = time()

        # Prediction results
        _, predicted = torch.max(logits, 1)
        pred_label = class_names[predicted.item()]
        # Calculate actual allocated layer counts
        rgb_layers = layer_allocation[0, 0, :].sum().item()
        depth_layers = layer_allocation[0, 1, :].sum().item()

    latency_ms = (end_time - start_time)*1000
    print("===> Used Layers - RGB:", int(rgb_layers), "Depth:", int(depth_layers))
    print("===> Result:", [class_names[i] for i in predicted.cpu().numpy()])
    return pred_label, rgb_layers, depth_layers, latency_ms

def take_pic(model=None, device=None, args=None):
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable streams (you can adjust resolution, format, and FPS as needed)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    plotter = LayerPlotter(max_history=30, height=200, width=640, total_layers=args.total_layers)
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
 
        if not color_frame or not depth_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        chart_img = plotter.draw()

        combined_camera = np.hstack((color_image, depth_colormap))
        chart_img_resized = cv2.resize(chart_img, (combined_camera.shape[1], 200))
        final_display = np.vstack((combined_camera, chart_img_resized))
        cv2.imshow('ADMN Real-time Demo', final_display)

        key = cv2.waitKey(1) & 0xFF

        
        # if now - last_capture_time >= 5.0:
        if key == ord('q'):

            print("==> Captured one RGB+Depth, sending to model...")
        
            data = transform_from_camera(color_image, depth_colormap)
            pred_label, rgb_layers, depth_layers, latency_ms= inference_stage2(model, data, device)
            # call update_led() to trigger the corresponding GPIO LED based on the predicted gesture 
            update_led(pred_label)
            # --------------------------------------------------------------------------------------
            plotter.update(rgb_layers, depth_layers)
            flops = conceptual_calculate_flops(
                total_layers=args.total_layers,
                rgb_layers_used=rgb_layers,
                depth_layers_used=depth_layers
            )
            print(f"===> Estimated FLOPs: {flops:.2f} GFLOPs")
            print("===> Latency:", latency_ms, "ms")
            updated_chart = plotter.draw()
            updated_chart_resized = cv2.resize(updated_chart, (combined_camera.shape[1], 200))
            
            final_display[480:, :] = updated_chart_resized 

            text = f"Pred: {pred_label} | RGB: {int(rgb_layers)} | Depth: {int(depth_layers)} | {latency_ms:.1f} ms"
            overlay = final_display.copy()
            cv2.rectangle(overlay, (0, 0), (1280, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, final_display, 0.4, 0, final_display)
            cv2.putText(final_display, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.imshow('ADMN Real-time Demo', final_display)
            cv2.waitKey(10)
    
        elif key == 27:  # ESC
            # print("Exit.")
            if led_1: led_1.off()
            if led_2: led_2.off()
            if led_3: led_3.off()
            if led_4: led_4.off()
            pipeline.stop()
            cv2.destroyAllWindows()
            # print("===> Avg Latency:", lat/10, "seconds")
            break

def main(args):
    # --- Setup and Model Loading ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nCreating model...")
    model = AdaptiveGestureClassifier(
        num_classes=4,
        adapter_hidden_dim=256,
        total_layers=args.total_layers,
        qoi_dim=128,
        stage1_checkpoint=None
    ).to(device)

    # Load the trained Stage 2 model weights
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Starting inference with Total Layers Budget: {args.total_layers}")
    model.eval()
    take_pic(model, device, args)


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
