"""
deploy_arm_terminal.py
----------------------
Terminal-controlled arm position classifier with rhythmic haptic feedback.
Each arm position has a unique vibration rhythm that feels like the position.

Haptic vocabulary (S = short buzz, L = long buzz):
  rest             ->  S          single soft tap
  overhead         ->  L          one long sustained buzz
  forward_90       ->  S S        two quick even taps
  side_90          ->  S L        short then long
  curl             ->  L S        long then short
  elbow_90_forward ->  S S S      three quick taps
  elbow_90_side    ->  S L S      short-long-short sandwich
  cross_body       ->  L L        two long buzzes

Hardware:
  - 3x MPU6050 via TCA9548A (ch1=shoulder, ch2=upper_arm, ch3=forearm)
  - Coin vibration motor on GPIO 21 via transistor
"""

import joblib
import pandas as pd
from time import sleep
from gpiozero import OutputDevice

from imu_reader_arm import setup, read_all_imus


# ----------------------------
# Configuration
# ----------------------------

MODEL_FILE      = "model_package_arm.joblib"
MOTOR_PIN       = 21

SHORT           = 0.15       # short buzz duration
LONG            = 0.45       # long buzz duration
GAP             = 0.12       # gap between pulses within a rhythm
SAMPLE_INTERVAL = 1.0        # seconds between reads in continuous mode


# ----------------------------
# Haptic rhythm definitions
# Each rhythm is a list of durations — the motor fires each in sequence
# with a GAP between them.
# ----------------------------

RHYTHMS = {
    "rest":             [SHORT],
    "overhead":         [LONG],
    "forward_90":       [SHORT, SHORT],
    "side_90":          [SHORT, LONG],
    "curl":             [LONG,  SHORT],
    "elbow_90_forward": [SHORT, SHORT, SHORT],
    "elbow_90_side":    [SHORT, LONG,  SHORT],
    "cross_body":       [LONG,  LONG],
}


# ----------------------------
# Globals
# ----------------------------

model        = None
feature_cols = None
encoder      = None
motor        = None


# ----------------------------
# Setup
# ----------------------------

def setup_hardware():
    """Initialize IMUs and coin motor."""
    global motor
    setup()
    motor = OutputDevice(MOTOR_PIN)
    print(f"Coin motor ready on GPIO {MOTOR_PIN}.\n")


def load_model():
    """Load model package saved by training_arm.py."""
    global model, feature_cols, encoder
    package      = joblib.load(MODEL_FILE)
    model        = package["model"]
    feature_cols = package["feature_cols"]
    encoder      = package["encoder"]
    print(f"Model loaded. Labels: {list(encoder.classes_)}\n")


# ----------------------------
# Motor helper
# ----------------------------

def fire_rhythm(label):
    """
    Look up the rhythm for label and fire each pulse in sequence.
    If label is not in RHYTHMS (shouldn't happen), fires a single short tap.
    """
    pattern = RHYTHMS.get(label, [SHORT])
    for i, duration in enumerate(pattern):
        motor.on()
        sleep(duration)
        motor.off()
        if i < len(pattern) - 1:    # gap between pulses, not after last one
            sleep(GAP)


# ----------------------------
# Inference
# ----------------------------

def predict_position():
    """
    Read all 3 IMUs, build feature row, return predicted label string.
    Returns None if any sensor read fails.
    """
    readings = read_all_imus()

    row = {}
    for col in feature_cols:
        angle  = col.split("_")[-1]
        seg    = "_".join(col.split("_")[:-1])
        angles = readings.get(seg)
        if angles is None:
            return None
        row[col] = angles[angle]

    X_row        = pd.DataFrame([row], columns=feature_cols)
    pred_encoded = model.predict(X_row)[0]
    return encoder.inverse_transform([pred_encoded])[0]


# ----------------------------
# Modes
# ----------------------------

def check_once():
    """Single snapshot — prints prediction and fires rhythm."""
    print("Reading... hold your position.\n")
    label = predict_position()

    if label is None:
        print("Sensor read failed. Check connections.\n")
        return

    rhythm_names = {SHORT: "S", LONG: "L"}
    pattern_str  = "  ".join(
        rhythm_names.get(d, "?") for d in RHYTHMS.get(label, [SHORT])
    )
    print(f"  Position : {label}")
    print(f"  Rhythm   : {pattern_str}")
    print()
    fire_rhythm(label)


def continuous_mode():
    """
    Reads every SAMPLE_INTERVAL seconds.
    Fires rhythm only when position changes — silent when stable.
    Ctrl+C to stop.
    """
    print("Continuous mode running. Ctrl+C to stop.\n")
    print(f"  {'Position':<22} {'Rhythm'}")
    print("  " + "-" * 36)

    last_label = None

    try:
        while True:
            label = predict_position()

            if label is None:
                print("  Sensor read failed — skipping.")
                sleep(SAMPLE_INTERVAL)
                continue

            if label != last_label:
                rhythm_names = {SHORT: "S", LONG: "L"}
                pattern_str  = "  ".join(
                    rhythm_names.get(d, "?") for d in RHYTHMS.get(label, [SHORT])
                )
                print(f"  {label:<22} {pattern_str}")
                fire_rhythm(label)
                last_label = label

            sleep(SAMPLE_INTERVAL)

    except KeyboardInterrupt:
        motor.off()
        print("\nStopped.\n")


def test_rhythms():
    """
    Fire every rhythm in sequence so you can learn them without
    needing the model. Prints the label before each fires.
    """
    print("\nFiring all rhythms in sequence. Learn to feel each one.\n")
    for label, pattern in RHYTHMS.items():
        rhythm_names = {SHORT: "S", LONG: "L"}
        pattern_str  = "  ".join(rhythm_names.get(d, "?") for d in pattern)
        print(f"  {label:<22} {pattern_str}")
        sleep(0.5)
        fire_rhythm(label)
        sleep(0.8)
    print()


# ----------------------------
# Menu
# ----------------------------

def print_menu():
    print("\nMenu:")
    print("  1) Initialize hardware + load model")
    print("  2) Check once (single snapshot)")
    print("  3) Continuous mode (Ctrl+C to stop)")
    print("  4) Test all rhythms (no model needed)")
    print("  5) Print haptic legend")
    print("  6) Quit")


def print_legend():
    print("\nHaptic legend  (S = short buzz, L = long buzz):\n")
    for label, pattern in RHYTHMS.items():
        rhythm_names = {SHORT: "S", LONG: "L"}
        pattern_str  = "  ".join(rhythm_names.get(d, "?") for d in pattern)
        print(f"  {label:<22} {pattern_str}")
    print()


def main():
    print("Arm Position Classifier — Terminal Mode\n")
    initialized = False

    while True:
        print_menu()
        choice = input("\nChoose (1-6): ").strip()

        if choice == "1":
            try:
                setup_hardware()
                load_model()
                initialized = True
            except FileNotFoundError:
                print(f"ERROR: {MODEL_FILE} not found. Run training_arm.py first.")
            except Exception as e:
                print(f"Setup error: {e}")

        elif choice in ("2", "3") and not initialized:
            print("Initialize first (option 1).")

        elif choice == "2":
            check_once()

        elif choice == "3":
            continuous_mode()

        elif choice == "4":
            if motor is None:
                try:
                    setup_hardware()
                except Exception as e:
                    print(f"Motor setup error: {e}")
                    continue
            test_rhythms()

        elif choice == "5":
            print_legend()

        elif choice == "6":
            if motor:
                motor.off()
            print("Done.")
            break

        else:
            print("Invalid choice. Pick 1-6.")


if __name__ == "__main__":
    main()