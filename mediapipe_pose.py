"""
mediapipe_pose.py
-----------------
Shoulder abduction angle and landmark confidence from Pi Camera via MediaPipe.

Adapted from Week 10 Lab 2. The live display loop is replaced with
get_pose_data() — a single function that returns numbers instead of
drawing on screen. Everything else (detector setup, angle math, landmark
extraction) is the same as the lab.

cv2display replaces cv2's display functions (imshow, waitKey,
destroyAllWindows) with pygame equivalents — required on this Pi setup
since GTK/X11 display is not available. Same patch used in the lab.

Landmark indices used:
  23 - LEFT_HIP      (base reference)
  11 - LEFT_SHOULDER (vertex — abduction angle measured here)
  13 - LEFT_ELBOW    (end of upper arm)

This gives true shoulder abduction angle — how far the arm has lifted
away from the body. The lab used elbow as vertex (elbow flexion angle)
which reads ~180 when the arm is straight out, not useful for abduction.

get_pose_data() returns:
  { "angle": float (degrees), "confidence": float (0.0 - 1.0) }
  or None if no pose detected.
"""

import cv2
import cv2display
cv2.imshow            = cv2display.imshow
cv2.waitKey           = cv2display.waitKey
cv2.destroyAllWindows = cv2display.destroyAllWindows

import mediapipe as mp
import numpy as np
from picamzero import Camera

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WINDOW_TITLE = "Pose Detection"
JOINT_A      = 23   # LEFT_HIP      (base reference point)
JOINT_B      = 11   # LEFT_SHOULDER (vertex — angle measured here)
JOINT_C      = 13   # LEFT_ELBOW    (end of upper arm)


# ---------------------------------------------------------------------------
# Detector setup — same as lab
# ---------------------------------------------------------------------------

def get_pose_detector():
    """Create and return a MediaPipe Pose detector."""
    return mp_pose.Pose(min_detection_confidence=0.5)


# ---------------------------------------------------------------------------
# Angle math — same as lab
# ---------------------------------------------------------------------------

def get_joint_angle(landmarks, a, b, c):
    """
    Angle in degrees at joint B formed by joints A, B, and C.

    Args:
        landmarks: pose landmarks object from MediaPipe
        a, b, c:   landmark indices (b is the vertex)

    Returns:
        float: angle in degrees (0-180)
    """
    landmark_a = landmarks.landmark[a]
    landmark_b = landmarks.landmark[b]
    landmark_c = landmarks.landmark[c]

    vec_ba = np.array([landmark_a.x - landmark_b.x, landmark_a.y - landmark_b.y])
    vec_bc = np.array([landmark_c.x - landmark_b.x, landmark_c.y - landmark_b.y])

    dot_product = np.dot(vec_ba, vec_bc)
    mag_ba      = np.linalg.norm(vec_ba)
    mag_bc      = np.linalg.norm(vec_bc)

    cos_angle = dot_product / (mag_ba * mag_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.degrees(np.arccos(cos_angle))


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def get_landmark_confidence(landmarks):
    """
    Average visibility score across the three landmarks we use.

    MediaPipe gives each landmark a visibility value (0.0 - 1.0).
    Averaging the three joints gives a rough signal for how much
    to trust the angle reading — used by fusion.py to weight
    MediaPipe vs IMU.
    """
    idxs   = [JOINT_A, JOINT_B, JOINT_C]
    scores = [landmarks.landmark[i].visibility for i in idxs]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Main data function — replaces run_pose_loop() from the lab
# ---------------------------------------------------------------------------

def get_pose_data(detector, frame):
    """
    Run pose detection on one frame and return angle + confidence.

    Args:
        detector: MediaPipe Pose object from get_pose_detector()
        frame:    BGR image array from the camera

    Returns:
        dict with "angle" (float) and "confidence" (float),
        or None if no pose is detected.
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = detector.process(image_rgb)

    if not results.pose_landmarks:
        return None

    angle      = get_joint_angle(results.pose_landmarks, JOINT_A, JOINT_B, JOINT_C)
    confidence = get_landmark_confidence(results.pose_landmarks)

    return {"angle": round(angle, 2), "confidence": round(confidence, 3)}


# ---------------------------------------------------------------------------
# Live display loop — same as lab, kept for testing
# ---------------------------------------------------------------------------

def run_pose_loop(cam):
    """
    Capture frames from the Pi Camera, detect body pose each frame,
    draw landmarks, display the shoulder abduction angle, and show the result.
    Press 'q' to quit.

    Always closes the camera and destroys windows on exit.

    Args:
        cam: an open picamzero Camera object
    """
    detector = get_pose_detector()
    try:
        while True:
            frame     = cam.capture_array()
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = detector.process(image_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )
                angle = get_joint_angle(
                    results.pose_landmarks, JOINT_A, JOINT_B, JOINT_C
                )
                cv2.putText(
                    frame, f"Shoulder: {angle:.1f} deg",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                )

            cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Menu — same pattern as lab
# ---------------------------------------------------------------------------

def print_menu():
    print("\nMenu:")
    print("  1) Test single frame (snapshot angle + confidence)")
    print("  2) Run live pose detection")
    print("  3) Quit")


def main():
    print("MediaPipe Pose — shoulder abduction angle\n")

    cam      = None
    detector = get_pose_detector()

    while True:
        print_menu()
        choice = input("\nChoose an option (1-3): ").strip()

        if choice == "1":
            try:
                cam   = Camera()
                frame = cam.capture_array()
                data  = get_pose_data(detector, frame)
                cam.close()
                cam = None
                if data:
                    print(f"\n  Angle:      {data['angle']:.1f} deg")
                    print(f"  Confidence: {data['confidence']:.3f}\n")
                else:
                    print("\nNo pose detected.\n")
            except Exception as e:
                print(f"\nCamera error: {e}\n")

        elif choice == "2":
            try:
                cam = Camera()
            except Exception as e:
                print(f"\nERROR: Cannot open camera: {e}\n")
                continue
            print(f"\nRunning live detection. Press 'q' to stop.\n")
            run_pose_loop(cam)
            cam = None
            print("\nDetection loop ended.\n")

        elif choice == "3":
            if cam is not None and hasattr(cam, "close"):
                cam.close()
            break

        else:
            print("\nInvalid choice. Pick 1-3.\n")

    print("\nDone.")


if __name__ == "__main__":
    main()