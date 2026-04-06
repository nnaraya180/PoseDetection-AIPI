"""
fusion.py
---------
Combines IMU and MediaPipe angle estimates into a single fused value.

Neither source is trusted unconditionally:
  - MediaPipe is accurate when landmarks are visible but blind to speed
    and unreliable when the arm moves out of frame.
  - IMUs are always available but drift slowly and have no body context.

The fused angle is a confidence-weighted average of the two. When
MediaPipe visibility drops below MIN_MP_CONFIDENCE, the IMUs take over
entirely. The gyro magnitude from the shoulder sensor feeds the speed
check independently — MediaPipe has no equivalent signal.

IMU angle proxy:
  Shoulder pitch is used as the abduction angle estimate from the IMUs.
  It's not a perfect match to the elbow angle MediaPipe measures, but
  it tracks the same movement and the model sees both anyway.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_MP_CONFIDENCE = 0.6     # below this, stop trusting MediaPipe angle
MAX_GYRO_DEG_S    = 60.0    # shoulder gyro above this flags as too fast


# ---------------------------------------------------------------------------
# Angle fusion
# ---------------------------------------------------------------------------

def fuse_angle(imu_readings, pose_data):
    """
    Weighted average of IMU and MediaPipe angle estimates.

    Args:
        imu_readings: dict from read_all_imus()
                      { segment: { "pitch", "roll", "gyro" } }
        pose_data:    dict from get_pose_data(), or None if no pose detected
                      { "angle": float, "confidence": float }

    Returns:
        fused angle in degrees (float), or None if IMU shoulder also failed.
    """
    shoulder = imu_readings.get("shoulder")
    imu_angle = shoulder["pitch"] if shoulder else None

    if imu_angle is None:
        return None

    if pose_data is None or pose_data["confidence"] < MIN_MP_CONFIDENCE:
        return imu_angle

    confidence = pose_data["confidence"]
    mp_angle   = pose_data["angle"]

    return round(confidence * mp_angle + (1.0 - confidence) * imu_angle, 2)


# ---------------------------------------------------------------------------
# Speed check
# ---------------------------------------------------------------------------

def is_too_fast(imu_readings):
    """
    True if the shoulder gyro exceeds MAX_GYRO_DEG_S.

    Uses shoulder only — it's the joint that drives the abduction movement.
    The model still sees all three gyro values as features; this flag is
    a separate pre-check used during data collection and deploy feedback.
    """
    shoulder = imu_readings.get("shoulder")
    if shoulder is None:
        return False
    return shoulder["gyro"] > MAX_GYRO_DEG_S


# ---------------------------------------------------------------------------
# Summary dict — what the rest of the pipeline consumes
# ---------------------------------------------------------------------------

def get_fusion_output(imu_readings, pose_data):
    """
    Single entry point for the deploy and data collection scripts.

    Args:
        imu_readings: from read_all_imus()
        pose_data:    from get_pose_data(), or None

    Returns:
        {
            "fused_angle":  float or None,
            "too_fast":     bool,
            "mp_angle":     float or None,
            "mp_confidence":float or None,
            "imu_angle":    float or None,   # shoulder pitch
        }
    """
    shoulder  = imu_readings.get("shoulder")
    imu_angle = shoulder["pitch"] if shoulder else None

    return {
        "fused_angle":   fuse_angle(imu_readings, pose_data),
        "too_fast":      is_too_fast(imu_readings),
        "mp_angle":      pose_data["angle"]      if pose_data else None,
        "mp_confidence": pose_data["confidence"] if pose_data else None,
        "imu_angle":     imu_angle,
    }


# ---------------------------------------------------------------------------
# Menu — lets you sanity check fusion without running the full deploy script
# ---------------------------------------------------------------------------

def print_menu():
    print("\nMenu:")
    print("  1) Print fusion config")
    print("  2) Quit")


def main():
    print("fusion.py — run from deploy_pt.py for live output.\n")
    print("This menu is for checking config values only.\n")

    while True:
        print_menu()
        choice = input("\nChoose (1-2): ").strip()

        if choice == "1":
            print(f"\n  MIN_MP_CONFIDENCE : {MIN_MP_CONFIDENCE}")
            print(f"  MAX_GYRO_DEG_S    : {MAX_GYRO_DEG_S} deg/s\n")

        elif choice == "2":
            print("Done.")
            break

        else:
            print("Invalid choice. Pick 1-2.")


if __name__ == "__main__":
    main()