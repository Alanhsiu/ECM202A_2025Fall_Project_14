import os
import sys
import torch
from tqdm import tqdm
import numpy as np
import pyrealsense2 as rs
import cv2
from time import time
from gpiozero import LED
from collections import Counter
from fvcore.nn import FlopCountAnalysis
import multiprocessing as mp
from multiprocessing import shared_memory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class_names = ['standing', 'left_hand', 'right_hand', 'both_hands']

PIC_INTERVAL = 5.0  

# Initialize GPIO LED Configuration ---
try:
    led_1 = LED(27)   # red LED GPIO 27
    led_2 = LED(22)   # green LED GPIO 22
    led_3 = LED(17)   # blue LED GPIO 17
    led_4 = LED(5)    # yellow LED GPIO 5
    led_5 = LED(6)
    print("[System] GPIO LEDs initialized.")
except Exception as e:
    print(f"[System] GPIO Init Failed: {e} (Ignore if not on RPi)")
    led_1 = None
    led_2 = None
    led_3 = None
    led_4 = None
    led_5 = None

# Update LED state by new predicted gesture ---
def update_led(label):
    if led_1 is None or led_2 is None or led_3 is None or led_4 is None or led_5 is None:
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

# Reset GPIO
def reset_gpio():
    if led_1: led_1.off()
    if led_2: led_2.off()
    if led_3: led_3.off()
    if led_4: led_4.off()
    if led_5: led_5.off()
    

def camera(shm_name, shape, dtype_str, num_slots,inference_start, inference_ended,result_queue,stop_event, camera_event, low_light_event, camera_ready, shared_lock, shared_state, log_queue):
    # shared memory setup (w/ inference process)
    dtype = np.dtype(dtype_str)
    shm = shared_memory.SharedMemory(name=shm_name)
    big_arr = np.ndarray((num_slots, *shape), dtype=dtype, buffer=shm.buf)

    rgb_target   = big_arr[0]
    depth_target = big_arr[1]

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable streams (you can adjust resolution, format, and FPS as needed)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    is_pipeline_active = False
    low_light = False
    camera_ready.set()
    log_queue.put("[Camera] Camera is ready.")

    camera_event.wait() # wait for the first trigger to start camera
    try:
        while True:
            if stop_event.is_set():
                break
            if not camera_event.is_set():
                if is_pipeline_active:
                    log_queue.put(">>> [Camera] Stopping Camera (Sleep Mode)...")
                    log_queue.put("=========================================================")
                    pipeline.stop()
                    led_5.off()
                    is_pipeline_active = False
                    reset_gpio()
                    with shared_lock:
                        shared_state["frame"] = None
                        shared_state["layer_rgb"] = None
                        shared_state["layer_depth"] = None
                        shared_state["last_result"] = None
                camera_event.wait() # wait until re-activated(interrupt from audio)
                continue
            elif is_pipeline_active == False:
                log_queue.put(">>> [Camera] Starting Camera...")
                log_queue.put("=========================================================")
                profile = pipeline.start(config)
                dd = profile.get_device()
                color_sensor = dd.first_color_sensor() 
                
                led_5.on()
                is_pipeline_active = True
                last_capture_time = time() - PIC_INTERVAL  # Force immediate capture on start

            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            combined_camera = np.hstack((color_image, depth_colormap))
            
            with shared_lock: # real-time camera frame update
                shared_state["frame"] = combined_camera.copy()

            if  low_light == False and low_light_event.is_set(): # low light mode
                log_queue.put(">>> [Camera] Low Light Mode Activated.")
                log_queue.put("=========================================================")
                low_light = True

                color_sensor.set_option(rs.option.enable_auto_exposure, 0) # disable auto exposure                 
                color_sensor.set_option(rs.option.exposure, 10.0) # set exposure 
                min_gain = color_sensor.get_option_range(rs.option.gain).min # set gain to min
                color_sensor.set_option(rs.option.gain, min_gain)
                continue

            if low_light and low_light_event.is_set() == False:
                log_queue.put(">>> [Camera] Exiting Low Light Mode.")
                low_light = False
                color_sensor.set_option(rs.option.enable_auto_exposure, 1) # enable auto exposure
                continue

            if inference_ended.is_set(): # inference result
                inference_ended.clear()
                pred_result = result_queue.get()
                pred_label = pred_result['Pred']
                rgb_layers = pred_result['RGB']
                depth_layers = pred_result['Depth']
                latency_ms = pred_result['latency_ms']
                # update LED
                update_led(pred_label)
                # terminal view
                log_queue.put(f"===> Prediction: {pred_label}")
                log_queue.put(f"===> Used Layers - RGB: {int(rgb_layers)} Depth: {int(depth_layers)}")
                log_queue.put(f"===> Latency: {latency_ms:.1f} ms")
                log_queue.put("=========================================================")
                
                # chart view
                with shared_lock:
                    shared_state["frame"] = combined_camera.copy()
                    shared_state["layer_rgb"] = rgb_layers
                    shared_state["layer_depth"] = depth_layers
                    shared_state["last_result"] = f"Pred: {pred_label} | RGB: {int(rgb_layers)} | Depth: {int(depth_layers)} | {latency_ms:.1f} ms"
        
            
            if time() - last_capture_time >= PIC_INTERVAL: # inference time
                
                log_queue.put("==> Captured one RGB+Depth, sending to model...")
                last_capture_time = time()

                rgb_target[:] = color_image
                depth_target[:] = depth_colormap
                inference_start.set()
                        
    finally:
        if is_pipeline_active:
            pipeline.stop()
        inference_start.set()  # in case inference is waiting
        reset_gpio()
    
