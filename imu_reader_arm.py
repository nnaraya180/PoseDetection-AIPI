"""
imu_reader_arm.py
-----------------
Reads pitch and roll from 3 MPU6050 IMUs via TCA9548A.

Channel mapping:
  ch 1 -> shoulder
  ch 2 -> upper_arm
  ch 3 -> forearm

Returns a dict: {segment_name: {"pitch": float, "roll": float}}
Same structure as all previous imu_reader files.
"""

import smbus2
import board
import busio
import adafruit_mpu6050
from math import atan2, sqrt, pi
from time import sleep


# ----------------------------
# Configuration
# ----------------------------

TCA_ADDRESS = 0x70

TCA_CHANNEL_MAP = {
    1: "shoulder",
    2: "upper_arm",
    3: "forearm",
}

SETTLE_SEC = 0.005      # 5ms settle after each channel switch


# ----------------------------
# Globals
# ----------------------------

i2c = None
bus = None


# ----------------------------
# Setup
# ----------------------------

def setup():
    """Initialize smbus2 (TCA switching) and busio (MPU6050 reads)."""
    global i2c, bus
    bus = smbus2.SMBus(1)
    i2c = busio.I2C(board.SCL, board.SDA)
    print("I2C buses initialized.")


# ----------------------------
# TCA helpers
# ----------------------------

def select_channel(channel):
    """Open one TCA channel. All others close."""
    bus.write_byte(TCA_ADDRESS, 1 << channel)
    sleep(SETTLE_SEC)


def close_channels():
    """Close all TCA channels."""
    bus.write_byte(TCA_ADDRESS, 0x00)
    sleep(SETTLE_SEC)


# ----------------------------
# Angle math
# ----------------------------

def compute_angles(ax, ay, az):
    """Return (pitch, roll) in degrees from raw accelerometer values."""
    pitch = atan2(-ax, sqrt(ay * ay + az * az)) * (180.0 / pi)
    roll  = atan2(ay, az) * (180.0 / pi)
    return round(pitch, 3), round(roll, 3)


# ----------------------------
# Single sensor read
# ----------------------------

def read_tca_imu(channel):
    """
    Open channel, read MPU6050, close channel.
    Returns {"pitch": float, "roll": float} or None on failure.
    """
    try:
        select_channel(channel)
        mpu = adafruit_mpu6050.MPU6050(i2c)
        ax, ay, az = mpu.acceleration
        pitch, roll = compute_angles(float(ax), float(ay), float(az))
        close_channels()
        return {"pitch": pitch, "roll": roll}
    except Exception as e:
        close_channels()
        print(f"Error reading channel {channel}: {e}")
        return None


# ----------------------------
# Read all 3 sensors
# ----------------------------

def read_all_imus():
    """
    Read all 3 arm IMUs in channel order (shoulder, upper_arm, forearm).
    Returns dict: {segment_name: {"pitch": float, "roll": float}}
    Value is None if that sensor failed.
    """
    readings = {}
    for channel, segment in TCA_CHANNEL_MAP.items():
        readings[segment] = read_tca_imu(channel)
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
    print("Arm IMU Reader\n")

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
            for segment, angles in readings.items():
                print(f"  {segment:<12}: {angles}")

        elif choice == "3":
            if i2c is None:
                print("Initialize first (option 1).")
                continue
            print("\nStreaming... Ctrl+C to stop.")
            print("Move your arm through each position to verify sensors respond.\n")
            try:
                while True:
                    readings = read_all_imus()
                    print("-" * 40)
                    for segment, angles in readings.items():
                        print(f"  {segment:<12}: {angles}")
                    sleep(0.5)
            except KeyboardInterrupt:
                print("\nStopped.\n")

        elif choice == "4":
            print("Done.")
            break

        else:
            print("Invalid choice. Pick 1-4.")


if __name__ == "__main__":
    main()