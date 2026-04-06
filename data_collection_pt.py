"""
data_collection_pt.py
---------------------
Records labeled movement state samples from 3 IMUs + MediaPipe.
Saves one CSV row per sample: timestamp, label, IMU features, fused angle.

Labels:
  rest         — arm at side, not moving
  moving_good  — arm raising or lowering at controlled speed
  peak_good    — arm held at ~90 deg abduction
  too_fast     — raising or lowering faster than a PT would want
  too_shallow  — arm raised but not reaching target range
  too_deep     — arm raised past 90 deg into painful range

Collection tips:
  Static labels (rest, peak_good, too_shallow, too_deep):
    Hold the position still for the full session, same as the midterm.

  Dynamic labels (moving_good, too_fast):
    Repeat the movement continuously for the full session.
    Aim for a steady rhythm — don't pause at top or bottom.

  too_fast:
    Same abduction movement but noticeably rushed.
    Watch the gyro column in imu_reader_pt.py stream mode first
    to get a feel for what fast looks like numerically.

  Collect 4-5 sessions per label across different sittings.
  Re-tape sensors in the same spot each time.
"""

import csv
import os
from time import sleep, time

from imu_reader_pt import setup, read_all_imus
from mediapipe_pose import get_pose_detector, get_pose_data
from fusion import get_fusion_output
from picamzero import Camera


# ----------------------------
# Configuration
# ----------------------------

CSV_FILE    = "pt_movement_data.csv"
DELAY_SEC   = 0.10          # 10 samples per second
CAPTURE_SEC = 30.0          # seconds per session

LABELS = [
    "rest",
    "moving_good",
    "peak_good",
    "too_fast",
    "too_shallow",
    "too_deep",
]

SEGMENTS = [
    "shoulder",
    "upper_arm",
    "forearm",
]

# 9 IMU features (pitch + roll + gyro per segment) + fused angle from fusion.py
CSV_HEADER = ["timestamp", "label"] + [
    f"{seg}_{val}"
    for seg in SEGMENTS
    for val in ("pitch", "roll", "gyro")
] + ["fused_angle"]


# ----------------------------
# CSV helpers — same as midterm
# ----------------------------

def write_header_if_needed(path):
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(CSV_HEADER)


def append_row(path, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


# ----------------------------
# Capture
# ----------------------------

def capture_session(label, cam, detector):
    """
    Sample all 3 IMUs + one camera frame at DELAY_SEC intervals for CAPTURE_SEC.
    Each sample is one CSV row.

    MediaPipe runs on every frame but only the fused angle makes it into
    the CSV — the raw mp_angle and confidence are used internally by
    get_fusion_output() and not stored separately.

    Returns number of samples saved.
    """
    write_header_if_needed(CSV_FILE)

    count = 0
    start = time()

    while time() - start < CAPTURE_SEC:
        imu_readings = read_all_imus()
        frame        = cam.capture_array()
        pose_data    = get_pose_data(detector, frame)
        fusion       = get_fusion_output(imu_readings, pose_data)
        timestamp    = round(time(), 4)

        row = [timestamp, label]

        for seg in SEGMENTS:
            data = imu_readings.get(seg)
            if data:
                row.append(data["pitch"])
                row.append(data["roll"])
                row.append(data["gyro"])
            else:
                row.extend([None, None, None])

        row.append(fusion["fused_angle"])
        append_row(CSV_FILE, row)
        count += 1
        sleep(DELAY_SEC)

    return count


# ----------------------------
# CSV summary — same structure as midterm
# ----------------------------

def show_csv_info():
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

    min_count = min(counts.values())
    if min_count < 100:
        short = [l for l, c in counts.items() if c < 100]
        print(f"\nWarning: {short} have fewer than 100 samples.")
        print("Aim for at least 3 full sessions per label.")


# ----------------------------
# Menu
# ----------------------------

def print_menu():
    print("\nMenu:")
    print("  1) Initialize sensors + camera")
    print("  2) Capture: rest         (arm at side, still)")
    print("  3) Capture: moving_good  (raise and lower, controlled)")
    print("  4) Capture: peak_good    (held at ~90 deg)")
    print("  5) Capture: too_fast     (same movement, rushed)")
    print("  6) Capture: too_shallow  (raised but short of target)")
    print("  7) Capture: too_deep     (past 90 deg)")
    print("  8) Show CSV info")
    print("  9) Quit")


def main():
    print("PT Movement — Data Collection")
    print(f"Each session: {CAPTURE_SEC:.0f}s at {1/DELAY_SEC:.0f} samples/sec\n")
    print("Static labels  : hold position still for the full session.")
    print("Dynamic labels : repeat the movement continuously.\n")

    initialized = False
    cam         = None
    detector    = None

    while True:
        print_menu()
        choice = input("\nChoose (1-9): ").strip()

        if choice == "1":
            try:
                setup()
                cam      = Camera()
                detector = get_pose_detector()
                initialized = True
                print("Sensors and camera ready.")
            except Exception as e:
                print(f"Setup error: {e}")

        elif choice in [str(i) for i in range(2, 8)] and not initialized:
            print("Initialize first (option 1).")

        elif choice in [str(i) for i in range(2, 8)]:
            label = LABELS[int(choice) - 2]
            print(f"\nLabel: {label}")
            print(f"Capturing for {CAPTURE_SEC:.0f} seconds...\n")
            try:
                n = capture_session(label, cam, detector)
                print(f"Saved {n} samples to {CSV_FILE}")
            except Exception as e:
                print(f"Capture error: {e}")

        elif choice == "8":
            show_csv_info()

        elif choice == "9":
            if cam:
                cam.close()
            print("Done.")
            break

        else:
            print("Invalid choice. Pick 1-9.")


if __name__ == "__main__":
    main()