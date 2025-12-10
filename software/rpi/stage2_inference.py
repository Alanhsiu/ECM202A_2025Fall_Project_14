import os
import sys
import torch
import numpy as np
import pyrealsense2 as rs
import cv2
from time import time
import multiprocessing as mp
from multiprocessing import shared_memory
from collections import Counter
from fvcore.nn import FlopCountAnalysis

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.adaptive_controller import AdaptiveGestureClassifier
from gesture_dataset import transform_from_camera

class_names = ['standing', 'left_hand', 'right_hand', 'both_hands']

G_BASE_FLOPS = 0.5e9
G_PER_LAYER_FLOPS = 1.0e9

# --- Calculate Base flops and per layer flops---
def calibrate_flops_constants(model, device):
    global G_BASE_FLOPS, G_PER_LAYER_FLOPS
    print(">>> [System] Calibrating FLOPs constants (Smart Mode)...")
    
    try:
        dummy_rgb = torch.randn(1, 3, 224, 224).to(device)
        dummy_depth = torch.randn(1, 3, 224, 224).to(device)

        model.eval()

        flops_analyzer = FlopCountAnalysis(model, (dummy_rgb, dummy_depth))
        flops_analyzer.unsupported_ops_warnings(False) 

        total_precise_flops = flops_analyzer.total()
        module_flops = flops_analyzer.by_module()

        candidates = []
        for name, flop in module_flops.items():
            if flop > total_precise_flops * 0.005: 
                candidates.append(flop)

        if not candidates:
            print(">>> [Warning] Calibration input too simple, model skipped all layers!")
            print(">>> [System] Fallback: Using default estimates.")
            G_BASE_FLOPS = 2.0e9
            G_PER_LAYER_FLOPS = 1.0e9
            return

        def round_to_significant(x, digits=2):
            if x == 0: return 0
            return round(x, -int(np.floor(np.log10(x))) + (digits - 1))

        grouped_flops = Counter([round_to_significant(c, 2) for c in candidates])

        most_common_flop_val, count = grouped_flops.most_common(1)[0]

        layer_costs = [c for c in candidates if round_to_significant(c, 2) == most_common_flop_val]
        avg_layer_cost = sum(layer_costs) / len(layer_costs)

        total_layers_cost = sum(layer_costs)

        calculated_base = total_precise_flops - total_layers_cost

        if calculated_base < 0: calculated_base = 0.1e9

        # uodate global variables for flops calculation
        G_BASE_FLOPS = calculated_base
        G_PER_LAYER_FLOPS = avg_layer_cost

        print(f">>> [System] Calibration Done!")
        # print(f"    - Precise Total: {total_precise_flops/1e9:.2f} GFLOPs")
        print(f"    - Detected {len(layer_costs)} active layers (avg cost: {avg_layer_cost/1e9:.2f} GFLOPs)")
        print(f"    - Calculated BASE: {G_BASE_FLOPS/1e9:.2f} GFLOPs")
        print(f"    - Calculated PER LAYER: {G_PER_LAYER_FLOPS/1e9:.2f} GFLOPs")

    except ImportError:
        print(">>> [Warning] 'fvcore' not installed. Using default FLOPs constants.")
    except Exception as e:
        print(f">>> [Warning] Calibration failed: {e}. Using default constants.")

# --- Conceptual FLOPs Calculation Function ---
def conceptual_calculate_flops(total_layers, rgb_layers_used, depth_layers_used):

    global G_BASE_FLOPS, G_PER_LAYER_FLOPS

    total_flops = G_BASE_FLOPS + \
                  (rgb_layers_used * G_PER_LAYER_FLOPS) + \
                  (depth_layers_used * G_PER_LAYER_FLOPS)

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
    
    return pred_label, rgb_layers, depth_layers, latency_ms

def inference_loop(shm_name, shape, dtype_str, num_slots, result_queue, inference_start, inference_ended, stop_event, model=None, device=None, args=None):
    dtype = np.dtype(dtype_str)
    shm = shared_memory.SharedMemory(name=shm_name)
    big_arr = np.ndarray((num_slots, *shape), dtype=dtype, buffer=shm.buf)

    rgb_target = big_arr[0]
    depth_target = big_arr[1]

    while True:
        if stop_event.is_set():
            break
        inference_start.wait()
        
        color_image = rgb_target.copy()
        depth_colormap = depth_target.copy()

        rgb_target[:] = 0
        depth_target[:] = 0
        data = transform_from_camera(color_image, depth_colormap)
        pred_label, rgb_layers, depth_layers, latency_ms= inference_stage2(model, data, device)
        
        # flops = conceptual_calculate_flops(
        #     total_layers=args.total_layers,
        #     rgb_layers_used=rgb_layers,
        #     depth_layers_used=depth_layers
        # )

        result_queue.put({'Pred': pred_label, 'RGB': rgb_layers, 'Depth': depth_layers, 'latency_ms': latency_ms})
        inference_start.clear()
        inference_ended.set()

def inference_init(shm_name, shape, dtype_str, num_slots, result_queue, inference_start, inference_ended, model_ready, stop_event, log_queue, args):
    # --- Setup and Model Loading ---
    os.sched_setaffinity(0, {1}) # Bind to CPU 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    log_queue.put("[Model] Creating model...")
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
    log_queue.put(f"[Model] Starting inference with Total Layers Budget: {args.total_layers}")
    model.eval()
    # calibrate_flops_constants(model, device)
    model_ready.set()
    log_queue.put("[Model] Model is ready.")

    inference_loop(shm_name, shape, dtype_str, num_slots, result_queue, inference_start, inference_ended, stop_event, model, device, args)

