"""
data_collection_arm.py
----------------------
Records labeled arm position snapshots from 3 IMUs.
Saves one CSV row per sample: timestamp, label, pitch/roll per segment.

Labels:
  rest             — arm down by side
  forward_90       — arm raised straight forward, elbow straight
  side_90          — arm raised out to the side, elbow straight
  curl             — bicep curl, hand up toward shoulder
  elbow_90_forward — upper arm forward, elbow bent 90 deg (forearm up)
  elbow_90_side    — upper arm to side, elbow bent 90 deg (forearm up)
  overhead         — arm fully raised overhead
  cross_body       — arm forward ~45 deg, elbow bent, forearm crosses chest

Collection tips:
  - Hold each position STILL for the full session.
  - Do 4-5 sessions per label across different sittings.
  - Re-tape sensors in the same spot each time (mark your arm).
  - Between sessions, let your arm fully rest before starting the next.
"""

import csv
import os
from time import sleep, time

from imu_reader_arm import setup, read_all_imus


# ----------------------------
# Configuration
# ----------------------------

CSV_FILE    = "arm_position_data.csv"
DELAY_SEC   = 0.10          # 10 samples per second
CAPTURE_SEC = 30.0          # seconds per session

LABELS = [
    "rest",
    "forward_90",
    "side_90",
    "curl",
    "elbow_90_forward",
    "elbow_90_side",
    "overhead",
    "cross_body",
]

# Column order — must stay consistent with training_arm.py
SEGMENTS = [
    "shoulder",
    "upper_arm",
    "forearm",
]

CSV_HEADER = ["timestamp", "label"] + [
    f"{seg}_{angle}"
    for seg in SEGMENTS
    for angle in ("pitch", "roll")
]


# ----------------------------
# CSV helpers
# ----------------------------

def write_header_if_needed(path):
    """Write CSV header only if file does not exist yet."""
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(CSV_HEADER)


def append_row(path, row):
    """Append one sample row to the CSV."""
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


# ----------------------------
# Capture
# ----------------------------

def capture_session(label):
    """
    Sample all 3 IMUs at DELAY_SEC intervals for CAPTURE_SEC seconds.
    Each sample is one CSV row: timestamp, label, pitch/roll per segment.
    Returns number of samples saved.
    """
    write_header_if_needed(CSV_FILE)

    count = 0
    start = time()

    while time() - start < CAPTURE_SEC:
        readings  = read_all_imus()
        timestamp = round(time(), 4)

        row = [timestamp, label]
        for seg in SEGMENTS:
            angles = readings.get(seg)
            if angles is not None:
                row.append(angles["pitch"])
                row.append(angles["roll"])
            else:
                row.append(None)
                row.append(None)

        append_row(CSV_FILE, row)
        count += 1
        sleep(DELAY_SEC)

    return count


# ----------------------------
# CSV summary
# ----------------------------

def show_csv_info():
    """Print total rows and per-label counts."""
    print(f"\nFile: {CSV_FILE}")
    if not os.path.exists(CSV_FILE):
        print("No data collected yet.")
        return

    counts = {label: 0 for label in LABELS}
    total  = 0

    with open(CSV_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            lbl = row.get("label", "")
            if lbl in counts:
                counts[lbl] += 1

    print(f"Total rows: {total}\n")
    print(f"{'Label':<20} {'Samples':>8}  {'Sessions (approx)':>18}")
    print("-" * 50)
    for lbl, cnt in counts.items():
        sessions = cnt / (CAPTURE_SEC / DELAY_SEC)
        print(f"  {lbl:<18} {cnt:>8}  {sessions:>14.1f}")

    # Warn if any label is underrepresented
    min_count = min(counts.values())
    if min_count < 100:
        short = [l for l, c in counts.items() if c < 100]
        print(f"\nWarning: {short} have fewer than 100 samples.")
        print("Aim for at least 3 full sessions (90+ samples) per label.")


# ----------------------------
# Menu
# ----------------------------

def print_menu():
    print("\nMenu:")
    print("  1) Initialize sensors")
    print("  2) Capture: rest             (arm down by side)")
    print("  3) Capture: forward_90       (arm raised forward)")
    print("  4) Capture: side_90          (arm raised to side)")
    print("  5) Capture: curl             (hand up to shoulder)")
    print("  6) Capture: elbow_90_forward (upper arm fwd, forearm up)")
    print("  7) Capture: elbow_90_side    (upper arm side, forearm up)")
    print("  8) Capture: overhead         (arm fully raised)")
    print("  9) Capture: cross_body       (arm crosses chest)")
    print(" 10) Show CSV info")
    print(" 11) Quit")


def main():
    print("Arm Position — Data Collection")
    print(f"Each session: {CAPTURE_SEC:.0f}s at {1/DELAY_SEC:.0f} samples/sec\n")
    print("Tip: collect 4-5 sessions per label. Hold each position still.")
    print("     Mark your arm so sensors go back in the same spot.\n")

    initialized = False

    while True:
        print_menu()
        choice = input("\nChoose (1-11): ").strip()

        if choice == "1":
            try:
                setup()
                initialized = True
                print("Sensors ready.")
            except Exception as e:
                print(f"Setup error: {e}")

        elif choice in [str(i) for i in range(2, 10)] and not initialized:
            print("Initialize sensors first (option 1).")

        elif choice in [str(i) for i in range(2, 10)]:
            label = LABELS[int(choice) - 2]
            print(f"\nLabel: {label}")
            print(f"Hold position. Capturing for {CAPTURE_SEC:.0f} seconds...\n")
            try:
                n = capture_session(label)
                print(f"Saved {n} samples to {CSV_FILE}")
            except Exception as e:
                print(f"Capture error: {e}")

        elif choice == "10":
            show_csv_info()

        elif choice == "11":
            print("Done.")
            break

        else:
            print("Invalid choice. Pick 1-11.")


if __name__ == "__main__":
    main()
