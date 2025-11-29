import os
import sys
import torch
from tqdm import tqdm
import argparse
import numpy as np
import pyrealsense2 as rs
import cv2
from time import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.adaptive_controller import AdaptiveGestureClassifier
from data.gesture_dataset import transform_from_camera

class_names = ['standing', 'left_hand', 'right_hand', 'both_hands']

# --- Layer Usage Plotter Class ---
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

# --- Inference Function ---
def inference_stage2(model, data, device, temperature=0.5):

    with torch.no_grad():
        rgb = data['rgb'].to(device)
        depth = data['depth'].to(device)

        # Measure model runtime (Latency)
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

def take_pic(camera_event, model=None, device=None, args=None):
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable streams (you can adjust resolution, format, and FPS as needed)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    # pipeline.start(config)
    plotter = LayerPlotter(max_history=30, height=200, width=640, total_layers=args.total_layers)
    
    is_pipeline_active = False
    camera_event.wait() # wait for the first trigger to start camera
    
    try:
        while True:
            if not camera_event.is_set():
                if is_pipeline_active:
                    print(">>> [Worker] Stopping Camera (Sleep Mode)...")
                    pipeline.stop()
                    is_pipeline_active = False
                    cv2.destroyAllWindows()
                    # TODO: Reset GPIO
                camera_event.wait()
                continue
            elif is_pipeline_active == False:
                print(">>> [Worker] Starting Camera...")
                pipeline.start(config)
                is_pipeline_active = True
                last_capture_time = time() - 5.0  # Force immediate capture on start

            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            chart_img = plotter.draw()

            combined_camera = np.hstack((color_image, depth_colormap))
            chart_img_resized = cv2.resize(chart_img, (combined_camera.shape[1], 200))
            final_display = np.vstack((combined_camera, chart_img_resized))
            cv2.imshow('ADMN Real-time Demo', final_display)

            key = cv2.waitKey(1) & 0xFF

            
            if time() - last_capture_time >= 5.0:
            # if key == ord('q'):

                print("==> Captured one RGB+Depth, sending to model...")
                last_capture_time = time()

                data = transform_from_camera(color_image, depth_colormap)
                pred_label, rgb_layers, depth_layers, latency_ms= inference_stage2(model, data, device)
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
        
            # elif key == 27:  # ESC
            #     pipeline.stop()
            #     cv2.destroyAllWindows()
            #     break
    finally:
        if is_pipeline_active:
            pipeline.stop()
        cv2.destroyAllWindows()
        # TODO: Reset GPIO

def camera(camera_event,camera_ready,args):
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
    camera_ready.set()
    take_pic(camera_event, model, device, args)


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

    camera(args)
