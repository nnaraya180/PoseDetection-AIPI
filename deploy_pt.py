"""
deploy_pt.py
------------
Live arm movement classifier with LCD feedback and rep counter.

Loads the trained model package, reads IMUs + camera each loop,
runs fusion, predicts movement state, and displays feedback on the LCD.

Starts automatically on boot. Button on GPIO 26 toggles between
standby and active mode — same pattern as Week 1 button lab.

LCD layout:
  Row 0: status message    (GOOD FORM / SLOW DOWN / TOO SHALLOW / TOO DEEP)
  Row 1: rep count + angle (Reps: 3   87 deg)

Rep counter logic:
  A complete rep requires this state sequence in order:
    rest -> moving_good -> peak_good -> moving_good -> rest
  Any quality-bad state during a rep is noted but the rep still counts.
  The sequence resets if rest is seen before the cycle completes.
"""

import joblib
import pandas as pd
import RPi.GPIO as GPIO
from time import sleep
from lcd_i2c import LCD_I2C
from picamzero import Camera

from imu_reader_pt import setup, read_all_imus
from mediapipe_pose import get_pose_detector, get_pose_data
from fusion import get_fusion_output


# ----------------------------
# Configuration
# ----------------------------

MODEL_FILE      = "model_package_pt.joblib"
LCD_ADDRESS     = 39
BUTTON_PIN      = 26
SAMPLE_INTERVAL = 0.10      # 10 Hz inference loop


# ----------------------------
# LCD messages per predicted label
# ----------------------------

# 16 chars max per row — checked against the 2x16 display
LCD_MESSAGES = {
    "rest":         "Ready",
    "moving_good":  "GOOD FORM",
    "peak_good":    "GOOD FORM",
    "too_fast":     "SLOW DOWN",
    "too_shallow":  "TOO SHALLOW",
    "too_deep":     "TOO DEEP",
}


# ----------------------------
# Rep counter
# ----------------------------

# Expected sequence for a complete rep — order matters
REP_SEQUENCE = [
    "rest",
    "moving_good",
    "peak_good",
    "moving_good",
    "rest",
]


class RepCounter:
    """
    Tracks movement state sequence and increments on a complete rep cycle.

    Watches for REP_SEQUENCE in order. Quality-bad states (too_fast,
    too_shallow, too_deep) don't break the sequence — the rep still
    counts, but the LCD message reflects the bad state while it's happening.
    """

    def __init__(self):
        self.reps = 0
        self.step = 0       # current position in REP_SEQUENCE

    def update(self, label):
        """
        Advance the sequence if label matches the next expected state.
        Resets to step 0 on unexpected rest (dropped rep).
        Returns True if a rep just completed.
        """
        expected = REP_SEQUENCE[self.step]

        if label == expected:
            self.step += 1
            if self.step == len(REP_SEQUENCE):
                self.reps += 1
                self.step = 0
                return True

        # Unexpected rest means the patient stopped mid-rep — reset
        elif label == "rest":
            self.step = 0

        # Quality-bad states pass through without breaking the sequence
        return False


# ----------------------------
# Globals
# ----------------------------

model        = None
feature_cols = None
encoder      = None
lcd          = None


# ----------------------------
# Setup
# ----------------------------

def setup_hardware():
    """Initialize IMUs and LCD."""
    setup()
    lcd_init()
    print("Hardware ready.\n")


def load_model():
    """Load model package saved by training_pt.py."""
    global model, feature_cols, encoder
    package      = joblib.load(MODEL_FILE)
    model        = package["model"]
    feature_cols = package["feature_cols"]
    encoder      = package["encoder"]
    print(f"Model loaded. Labels: {list(encoder.classes_)}\n")


# ----------------------------
# LCD helpers — from Week 1 lab
# ----------------------------

def lcd_init():
    """Create LCD object and turn on backlight."""
    global lcd
    lcd = LCD_I2C(LCD_ADDRESS, 16, 2)
    lcd.backlight.on()


def lcd_write(row0, row1=""):
    """Write up to two lines. Pads to 16 chars to clear leftover text."""
    lcd.cursor.setPos(0, 0)
    lcd.write_text(row0[:16].ljust(16))
    lcd.cursor.setPos(1, 0)
    lcd.write_text(row1[:16].ljust(16))


# ----------------------------
# Button helper
# ----------------------------

def button_pressed():
    """Debounced button read — returns True only if held for 50ms."""
    if GPIO.input(BUTTON_PIN) == GPIO.LOW:
        sleep(0.05)
        return GPIO.input(BUTTON_PIN) == GPIO.LOW
    return False


# ----------------------------
# Inference
# ----------------------------

def predict_state(imu_readings, pose_data):
    """
    Build feature row from fusion output and return predicted label string.
    Returns None if any required sensor data is missing.
    """
    fusion = get_fusion_output(imu_readings, pose_data)

    row = {}
    for col in feature_cols:
        if col == "fused_angle":
            if fusion["fused_angle"] is None:
                return None
            row[col] = fusion["fused_angle"]
        else:
            parts = col.split("_")
            val   = parts[-1]               # pitch / roll / gyro
            seg   = "_".join(parts[:-1])    # shoulder / upper_arm / forearm
            data  = imu_readings.get(seg)
            if data is None:
                return None
            row[col] = data[val]

    X_row        = pd.DataFrame([row], columns=feature_cols)
    pred_encoded = model.predict(X_row)[0]
    return encoder.inverse_transform([pred_encoded])[0]


# ----------------------------
# Active session loop
# ----------------------------

def run_session(cam, detector):
    """
    Inference loop for one active session.
    Runs until the button is pressed again to return to standby.
    """
    counter = RepCounter()
    lcd_write("Starting...", "")
    sleep(1)

    print("Session active. Press button to return to standby.\n")
    print(f"  {'State':<20} {'Reps':>5}  {'Angle':>8}")
    print("  " + "-" * 38)

    while True:
        if button_pressed():
            break

        imu_readings = read_all_imus()
        frame        = cam.capture_array()
        pose_data    = get_pose_data(detector, frame)
        fusion       = get_fusion_output(imu_readings, pose_data)

        label = predict_state(imu_readings, pose_data)

        if label is None:
            lcd_write("SENSOR ERROR", "Check connections")
            sleep(SAMPLE_INTERVAL)
            continue

        counter.update(label)

        angle_str = (
            f"{fusion['fused_angle']:.0f} deg"
            if fusion["fused_angle"] is not None
            else "--"
        )
        row1 = f"Reps:{counter.reps:<3}  {angle_str}"
        lcd_write(LCD_MESSAGES.get(label, label), row1)

        print(f"  {label:<20} {counter.reps:>5}  {angle_str:>8}")
        sleep(SAMPLE_INTERVAL)

    print(f"\nSession ended. Total reps: {counter.reps}\n")
    lcd_write("Standby", "Press to start")
    sleep(0.5)      # debounce — prevent immediate re-trigger


# ----------------------------
# Main
# ----------------------------

def main():
    """
    Auto-initializes on boot, then waits for button press to start a session.
    Button on GPIO 26 toggles between standby and active — same pattern
    as the Week 1 button lab.
    """
    print("PT Movement Classifier — initializing...\n")

    setup_hardware()
    load_model()
    cam      = Camera()
    detector = get_pose_detector()

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    lcd_write("PT Assistant", "Press to start")
    print("Ready. Press button to start a session.\n")

    try:
        while True:
            if button_pressed():
                run_session(cam, detector)
            sleep(0.05)     # idle poll — keeps CPU quiet while waiting

    except KeyboardInterrupt:
        if cam:
            cam.close()
        lcd_write("", "")
        lcd.backlight.off()
        GPIO.cleanup()
        print("\nDone.")


if __name__ == "__main__":
    main()