import subprocess
import numpy as np
import cv2

WIDTH, HEIGHT = 640, 480

class Camera:
    def __init__(self):
        self.proc = subprocess.Popen(
            ["rpicam-vid", "-t", "0", "--width", str(WIDTH),
             "--height", str(HEIGHT), "--codec", "yuv420",
             "--output", "-", "--nopreview"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        self.frame_size = WIDTH * HEIGHT * 3 // 2

    def capture_array(self):
        raw = self.proc.stdout.read(self.frame_size)
        if len(raw) != self.frame_size:
            raise RuntimeError("Failed to capture frame")
        yuv = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT * 3 // 2, WIDTH))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    def close(self):
        self.proc.terminate()
        self.proc.wait()
