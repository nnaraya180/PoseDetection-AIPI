# PoseDetection-AIPI

# Dual-Modality Physical Therapy Exercise Coach
**AIPI 590 — Final Individual Project**

---

## Project Overview

This project extends the midterm IMU-based arm position classifier into a real-time physical therapy coaching system for shoulder abduction exercises. Three MPU-6050 IMUs mounted on the forearm, upper arm, and shoulder are fused with MediaPipe pose estimation from a Pi Camera to classify movement quality and deliver corrective feedback through an LCD display.

The motivation for combining two sensor modalities comes from their complementary failure modes. MediaPipe is accurate when the body is clearly visible but loses reliability when landmarks are occluded or at extreme joint angles. IMUs are always available and capture angular velocity — something a camera fundamentally cannot measure — but accumulate drift over time. The fusion layer in this system uses MediaPipe confidence scores to weight how much each source contributes to the angle estimate on any given frame, letting the stronger signal lead.

---

## Motivation

Physical therapy adherence outside clinical settings is a known problem. Patients doing home exercises without supervision tend to develop movement compensations that reduce efficacy or cause injury. This project is a proof-of-concept for a low-cost coaching device that provides real-time corrective feedback between appointments — no internet connection, no cloud inference, everything runs locally on the Pi.

---

## Hardware

| Component | Role |
|---|---|
| Raspberry Pi 4 | Primary compute |
| 3x MPU-6050 (GY-521) via TCA9548A | IMUs on forearm, upper arm, shoulder |
| Pi Camera | Shoulder abduction angle via MediaPipe |
| 16x2 I2C LCD | Live feedback and rep counter |
| Button (GPIO 26) | Start / stop session |

---

## What the Model Predicts

A Random Forest classifier takes 10 features as input — pitch, roll, and gyro magnitude from each of the three IMU segments (9 features) plus a fused angle estimate from the MediaPipe + IMU weighted average (1 feature) — and classifies each frame into one of six movement states:

| Label | Meaning |
|---|---|
| `rest` | Arm at side, not moving |
| `moving_good` | Raising or lowering at a controlled pace |
| `peak_good` | Arm held at ~90° abduction |
| `too_fast` | Movement speed exceeds safe threshold |
| `too_shallow` | Arm raised but not reaching target range |
| `too_deep` | Arm raised past 90° into uncomfortable range |

The rep counter watches for the sequence `rest → moving_good → peak_good → moving_good → rest` and increments on each complete cycle. Quality-bad states during a rep are flagged on the LCD but do not break the count.

---

## IMU Drift

IMU drift was a real concern going into this project, particularly for the gyroscope. The MPU-6050 gyro accumulates error over time as small offsets integrate — this is why dead reckoning with a gyro alone is unreliable for anything longer than a few seconds.

This system addresses drift in two ways:

**Short-term: rolling average smoothing.** Each sensor's gyro magnitude is passed through a 5-sample rolling average (~0.5 seconds at 10Hz) before being handed to the model. This cuts the high-frequency noise that makes drift worse and keeps the feature values stable during still periods.

**Structural: the model uses pitch and roll from the accelerometer, not integrated gyro angle.** Accelerometer-derived pitch and roll do not drift — they are calculated fresh from gravity on every read. The gyro magnitude feature is used only to detect movement speed (`too_fast`), not to track position over time. This sidesteps the main drift problem entirely.

**Fusion as anchor.** When MediaPipe confidence is above the threshold, the fused angle estimate is pulled toward the camera-derived measurement each frame. This acts as a soft correction — if the IMU angle has wandered slightly, a high-confidence MediaPipe frame pulls it back. When the arm is occluded and confidence drops below 0.6, the system falls back to IMU-only, which is acceptable for the short windows where occlusion typically occurs.

---

## System Pipeline

```
Camera ──► MediaPipe Pose ──► Shoulder angle + confidence ──┐
                                                             ├──► fusion.py (weighted average)
3x MPU-6050 ──► imu_reader_pt.py ──► pitch, roll, gyro ────┘
                                                             │
                                                    Random Forest classifier
                                                             │
                                                    LCD: form cue + rep count
```

---

## Project Structure

```
PoseDetection-AIPI/
│
├── imu_reader_pt.py        # Reads pitch, roll, gyro from 3 IMUs via TCA9548A
├── mediapipe_pose.py       # Shoulder abduction angle + landmark confidence
├── fusion.py               # Weighted angle estimate from IMU + MediaPipe
├── data_collection_pt.py   # Labeled session capture → CSV
├── training_pt.py          # Random Forest training pipeline
├── deploy_pt.py            # Live inference loop with LCD + button control
│
├── picamzero.py            # Custom camera shim (rpicam-vid subprocess)
├── cv2display.py           # pygame display shim (replaces cv2 GUI)
├── lcd_i2c.py              # LCD shim wrapping i2c_lcd for Python 3.7
│
├── pt_movement_data.csv    # Collected training data
├── model_package_pt.joblib # Trained model, encoder, feature columns
└── README.md
```

---

## How to Run

This project runs on Python 3.7 specifically. Do not use `python3` — on this Pi that resolves to 3.11 which has no mediapipe wheel for armv7l.

### Collect training data

```bash
python3.7 data_collection_pt.py
```

Work through each label. Static labels (rest, peak_good, too_shallow, too_deep) — hold the position for the full 30 second session. Dynamic labels (moving_good, too_fast) — repeat the movement continuously for 30 seconds. Aim for 300+ samples per label before training.

### Train the model

```bash
python3.7 training_pt.py
```

Run steps 1-7 in order. Check the confusion matrix (option 6) before saving — `moving_good` and `too_fast` are the most likely to bleed into each other given they are the same motion at different speeds.

### Deploy

```bash
python3.7 deploy_pt.py
```

Initializes automatically. Press the button on GPIO 26 to start a session, press again to return to standby.

---

## Model Results

Training on ~1,200 samples across 6 labels:

- Overall accuracy: 97.9%
- `rest`, `too_shallow`, `too_deep`, `peak_good`: near-perfect classification
- `too_fast`: 89% — weakest class, limited training data
- Top features: `forearm_pitch`, `upper_arm_pitch`, `upper_arm_gyro`
- `fused_angle` contributed 6.4% importance, confirming the MediaPipe fusion layer adds signal beyond raw IMU alone

---

## Challenges

**Python 3.7 on armv7l.** The only available mediapipe wheel for this Pi's architecture targets Python 3.7, which required compiling Python from source. Several shared libraries needed symlinks or stubs to resolve binary compatibility issues with newer system versions.

**IMU drift on extended sessions.** Addressed by using accelerometer-derived angles for position (no drift) and reserving gyro only for speed detection. The fusion layer also provides soft correction from MediaPipe on frames where landmarks are visible.

**Camera capture reliability.** The rpicam-vid subprocess occasionally drops frames during long sessions. The capture loop handles this gracefully — failed frames fall back to IMU-only fusion rather than crashing.

**Separating ascending from descending.** Initial design had separate labels for raising and lowering. In practice, the IMU gyro magnitude is identical for both directions — only a signed gyro axis would distinguish them. Collapsed to a single `moving_good` label, which is sufficient for rep counting and form assessment.

---

## Next Steps

- Collect data from multiple people to generalize beyond a single user
- Add signed gyro axis as a feature to separate raising from lowering
- Expand to additional PT exercises (lateral raise, external rotation)
- Session logging to track progress over time

---

## References

- MediaPipe Pose: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
- MPU-6050 datasheet
- AIPI 590 course materials, Duke University
