"""
imu_reader_pt.py
----------------
pitch, roll, and angular velocity from 3 MPU6050s via TCA9548A.

Extended from imu_reader_arm.py. Main addition is gyro magnitude
per sensor — needed to catch too-fast reps at the model level.
Gyro is smoothed with a short rolling window to cut idle noise.

All values in degrees / deg/s so units are consistent across features.

Channel map:
  ch 1 -> shoulder
  ch 2 -> upper_arm
  ch 3 -> forearm

read_all_imus() returns:
  { segment: { "pitch": float, "roll": float, "gyro": float } }
"""

import smbus2
import board
import busio
import adafruit_mpu6050
from math import atan2, sqrt, pi
from time import sleep
from collections import deque


# ----------------------------
# Configuration
# ----------------------------

TCA_ADDRESS = 0x70

TCA_CHANNEL_MAP = {
    1: "shoulder",
    2: "upper_arm",
    3: "forearm",
}

SETTLE_SEC  = 0.005     # TCA needs a moment after switching channels
GYRO_WINDOW = 5         # rolling average size — 5 samples at 10 Hz ~ 0.5s


# ----------------------------
# Globals
# ----------------------------

i2c = None
bus = None
_gyro_windows = {}      # one deque per segment, initialized in setup()


# ----------------------------
# Setup
# ----------------------------

def setup():
    """Open both I2C buses and create a gyro smoothing window per sensor."""
    global i2c, bus, _gyro_windows
    bus = smbus2.SMBus(1)
    i2c = busio.I2C(board.SCL, board.SDA)
    for segment in TCA_CHANNEL_MAP.values():
        _gyro_windows[segment] = deque(maxlen=GYRO_WINDOW)
    print("I2C buses initialized.")


# ----------------------------
# TCA helpers
# ----------------------------

def select_channel(channel):
    """Activate one multiplexer channel; the others go quiet."""
    bus.write_byte(TCA_ADDRESS, 1 << channel)
    sleep(SETTLE_SEC)


def close_channels():
    """Deactivate all multiplexer channels."""
    bus.write_byte(TCA_ADDRESS, 0x00)
    sleep(SETTLE_SEC)


# ----------------------------
# Angle math
# ----------------------------

def compute_angles(ax, ay, az):
    """Pitch and roll in degrees from raw accelerometer output."""
    pitch = atan2(-ax, sqrt(ay * ay + az * az)) * (180.0 / pi)
    roll  = atan2(ay, az) * (180.0 / pi)
    return round(pitch, 3), round(roll, 3)


# ----------------------------
# Gyro math
# ----------------------------

def compute_gyro_magnitude(gx, gy, gz):
    """
    Total rotation rate in deg/s.

    mpu.gyro comes back in rad/s. Converting to deg/s keeps units
    consistent with pitch and roll. Collapsing to magnitude because
    the model cares about overall movement speed, not which axis.
    """
    gx_deg = gx * (180.0 / pi)
    gy_deg = gy * (180.0 / pi)
    gz_deg = gz * (180.0 / pi)
    return round(sqrt(gx_deg**2 + gy_deg**2 + gz_deg**2), 3)


def smooth_gyro(segment, raw_magnitude):
    """Running average over the last GYRO_WINDOW readings for one sensor."""
    window = _gyro_windows[segment]
    window.append(raw_magnitude)
    return round(sum(window) / len(window), 3)


# ----------------------------
# Single sensor read
# ----------------------------

def read_tca_imu(channel, segment):
    """
    Read accel and gyro from one sensor in a single channel-open window.

    Returns {"pitch": float, "roll": float, "gyro": float} or None on failure.
    """
    try:
        select_channel(channel)
        mpu = adafruit_mpu6050.MPU6050(i2c)

        ax, ay, az = mpu.acceleration
        pitch, roll = compute_angles(float(ax), float(ay), float(az))

        gx, gy, gz = mpu.gyro
        raw_gyro      = compute_gyro_magnitude(float(gx), float(gy), float(gz))
        smoothed_gyro = smooth_gyro(segment, raw_gyro)

        close_channels()
        return {"pitch": pitch, "roll": roll, "gyro": smoothed_gyro}

    except Exception as e:
        close_channels()
        print(f"Error reading channel {channel}: {e}")
        return None


# ----------------------------
# Read all 3 sensors
# ----------------------------

def read_all_imus():
    """
    Poll all three sensors and return one dict keyed by segment name.

    Any sensor that fails comes back as None — callers should check.
    """
    readings = {}
    for channel, segment in TCA_CHANNEL_MAP.items():
        readings[segment] = read_tca_imu(channel, segment)
    return readings


# ----------------------------
# Menu
# ----------------------------

def print_menu():
    print("\nMenu:")
    print("  1) Initialize I2C buses")
    print("  2) Read all 3 IMUs (single snapshot)")
    print("  3) Stream all 3 IMUs (Ctrl+C to stop)")
    print("  4) Quit")


def main():
    print("PT IMU Reader — pitch, roll + gyro\n")

    while True:
        print_menu()
        choice = input("\nChoose (1-4): ").strip()

        if choice == "1":
            try:
                setup()
            except Exception as e:
                print(f"Setup error: {e}")

        elif choice == "2":
            if i2c is None:
                print("Initialize first (option 1).")
                continue
            readings = read_all_imus()
            print()
            for segment, data in readings.items():
                if data:
                    print(f"  {segment:<12}: pitch={data['pitch']:>8.2f}°  "
                          f"roll={data['roll']:>8.2f}°  "
                          f"gyro={data['gyro']:>7.2f} deg/s")
                else:
                    print(f"  {segment:<12}: READ FAILED")

        elif choice == "3":
            if i2c is None:
                print("Initialize first (option 1).")
                continue
            print("\nStreaming... Ctrl+C to stop.")
            print("Move your arm slowly then fast — watch the gyro column.\n")
            try:
                while True:
                    readings = read_all_imus()
                    print("-" * 60)
                    for segment, data in readings.items():
                        if data:
                            print(f"  {segment:<12}: pitch={data['pitch']:>8.2f}°  "
                                  f"roll={data['roll']:>8.2f}°  "
                                  f"gyro={data['gyro']:>7.2f} deg/s")
                        else:
                            print(f"  {segment:<12}: READ FAILED")
                    sleep(0.1)
            except KeyboardInterrupt:
                print("\nStopped.\n")

        elif choice == "4":
            print("Done.")
            break

        else:
            print("Invalid choice. Pick 1-4.")


if __name__ == "__main__":
    main()