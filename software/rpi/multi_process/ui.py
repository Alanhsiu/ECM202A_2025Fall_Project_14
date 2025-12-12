import time
import cv2
import numpy as np
import queue
from collections import deque
import psutil
class CpuPlotter:
    def __init__(self, max_history=60, height=200, width=640):
        self.max_history = max_history
        self.height = height
        self.width = width

        self.bg_color = (20, 20, 20)
        self.line_color = (0, 255, 0)
        self.grid_color = (50, 50, 50)
        self.text_color = (255, 255, 255)

        # initalize with current cpu percent
        self.history = deque([psutil.cpu_percent(interval=None)] * max_history, maxlen=max_history) 

    def update(self):
        cpu_percent = psutil.cpu_percent(interval=None)
        self.history.append(cpu_percent)
        return cpu_percent

    def draw(self):
        canvas = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)

        # 0,25,50,75,100% grid + label
        for p in [0, 25, 50, 75, 100]:
            y = self.height - int(p / 100.0 * (self.height - 20)) - 10
            cv2.line(canvas, (0, y), (self.width, y), self.grid_color, 1)
            cv2.putText(
                canvas,
                f"{p}%",
                (5, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                self.grid_color,
                1,
            )

        step_x = self.width / (self.max_history - 1)
        pts = []
        for i, val in enumerate(self.history):
            x = int(i * step_x)
            y = self.height - int(val / 100.0 * (self.height - 20)) - 10
            pts.append((x, y))

        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i + 1], self.line_color, 2)

        last_val = self.history[-1]
        cv2.putText(
            canvas,
            f"CPU: {last_val:.1f}%",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.text_color,
            1,
        )

        return canvas
    
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
    
class ConsolePanel:
    def __init__(self, width=1280, height=120, max_lines=6):
        self.width = width
        self.height = height
        self.max_lines = max_lines

        self.bg_color = (0, 0, 0)          
        self.text_color = (0, 255, 0)      
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.thickness = 1
        self.line_spacing = 20 

        self.lines = deque([], maxlen=max_lines)

    def log(self, msg:str):
        # add new log line with timestamp
        ts = time.strftime("%H:%M:%S")  
        line = f"[{ts}] {msg}"

        MAX_CHAR = 100
        if len(line) > MAX_CHAR:
            line = line[:MAX_CHAR - 3] + "..."

        self.lines.append(line)

    def clear(self):
        # clear console
        self.lines.clear()

    def draw(self):
        # return an image of shape (height, width, 3) that can be directly pasted to UI layout
        canvas = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)

        y = 20  # 第一行文字位置
        for line in list(self.lines):
            cv2.putText(
                canvas,
                line,
                (10, y),
                self.font,
                self.font_scale,
                self.text_color,
                self.thickness,
                cv2.LINE_AA
            )
            y += self.line_spacing

        return canvas
    
class camera_ui:
    def __init__(self):
        self.BLACK_CAMERA = np.zeros((480, 1280, 3), dtype=np.uint8)
        
        text = "NO IMAGE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness = 3
        color = (255, 255, 255)

        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # position to center the text
        x = (self.BLACK_CAMERA.shape[1] - text_width) // 2
        y = (self.BLACK_CAMERA.shape[0] + text_height) // 2

        # draw the text
        cv2.putText(self.BLACK_CAMERA, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    def draw(self):
        return self.BLACK_CAMERA.copy()

def ui_generate(camera_event, stop_event, low_light_event, shared_state, shared_lock, log_queue, total_layers = 12):
    cpu_plotter = CpuPlotter(max_history=60, height=200, width=640)
    layer_plotter = LayerPlotter(max_history=30, height=200, width=640, total_layers=total_layers)
    console = ConsolePanel(width=1280, height=200, max_lines=6)
    placeholder_camera = camera_ui()
    
    cpu_time = time.time()
    last_camera_frame = None
    last_result = None
    while not stop_event.is_set():
        # 1) log_queue
        try:
            while True:
                msg = log_queue.get_nowait()
                console.log(msg)
        except queue.Empty:
            pass

        # 2) Update cpu every second
        if (time.time() - cpu_time) >= 1.0:
            cpu_time = time.time()
            cpu_plotter.update()
        cpu_chart = cpu_plotter.draw()
        
        # 3) Update layer/camera
        if not camera_event.is_set(): # camera not opened yet
            camera_frame = placeholder_camera.draw()
            layer_img = layer_plotter.draw()
        else:
            with shared_lock:
                layer_rgb = shared_state["layer_rgb"]
                layer_depth = shared_state["layer_depth"]
                frame = shared_state["frame"]
                last_result = shared_state["last_result"] 
                if layer_rgb is not None and layer_depth is not None:
                    shared_state["layer_rgb"] = None
                    shared_state["layer_depth"] = None

            # Update camera frame
            if frame is None and last_camera_frame is None:
                camera_frame = placeholder_camera.draw()
            elif frame is None and last_camera_frame is not None:
                camera_frame = last_camera_frame.copy()
            else:
                camera_frame = frame.copy()
                last_camera_frame = camera_frame.copy()

            # Update layer plotter if layer information is available
            if layer_rgb is not None and layer_depth is not None:
                layer_plotter.update(layer_rgb, layer_depth)

            layer_img = layer_plotter.draw()

        chart_left  = cv2.resize(layer_img, (640, 200))
        chart_right = cv2.resize(cpu_chart,   (640, 200))
        charts_row  = np.hstack((chart_left, chart_right))

        console_img = console.draw()

        final_display = np.vstack((camera_frame, charts_row, console_img))

        # 4) Summary bar
        if last_result is not None:
            text = last_result
            overlay = final_display.copy()
            cv2.rectangle(overlay, (0, 0), (final_display.shape[1], 40), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, final_display, 0.4, 0, final_display)
            cv2.putText( final_display, text, (20, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        scale = 2
        display_small = cv2.resize(final_display, None, fx=scale, fy=scale)
        cv2.imshow("ADMN Dashboard", display_small)
        key = cv2.waitKey(100) & 0xFF

        if key == 27:  # ESC
            stop_event.set()
            camera_event.set()
            cv2.destroyAllWindows()
            break
        if(key == ord('c')):  
            console.clear()
        if(key == ord('l')):  
            low_light_event.set()
        if(key == ord('n')):  
            low_light_event.clear()

