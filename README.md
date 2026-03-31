# PoseDetection-AIPI

# Dual-Modality Physical Therapy Exercise Coach
**AIPI 590 — Final Individual Project**

---

## Project Overview

This project extends the midterm IMU-based limb orientation classifier into a real-time physical therapy exercise coaching system. By fusing two complementary sensor modalities — a wrist-mounted IMU and a camera-based pose detection model — the system classifies exercise form quality during a single target exercise (bicep curl) and delivers immediate corrective feedback through visual and physical outputs.

The core insight motivating this design is that camera-based pose detection and IMU sensors have complementary failure modes. Pose detection loses precision at fine joint angles and degrades when limbs are partially occluded. IMUs are view-independent and capture precise angular velocity and orientation, but lack global body context. Together, they create a more robust and clinically meaningful signal than either sensor alone.

---

## Motivation and Background

Physical therapy adherence is a well-documented challenge. Patients performing home exercises without supervision frequently develop incorrect movement patterns that reduce efficacy or cause injury. A low-cost, Raspberry Pi-based coaching device could provide real-time corrective feedback between clinical appointments. This project demonstrates a proof-of-concept pipeline for that use case.

---

## Sensors and Hardware

| Component | Role |
|---|---|
| Raspberry Pi 4 | Primary compute platform |
| IMU (MPU-6050) | Wrist-mounted; captures arm orientation and angular velocity |
| USB Camera | Full-body pose detection |
| LCD Screen (16x2 or SPI) | Physical user interface — rep count, form grade, cues |
| LEDs (Green / Red) | Immediate binary form feedback |
| Optional: Servo motor | Analog "form score" dial |

---

## What the Model Predicts

A TFLite classification model (built on the midterm's IMU pipeline) takes fused features from both the IMU and pose keypoints as input and classifies each phase of a rep into one of three states:

- **CORRECT** — joint angle and body posture are within acceptable range
- **SHALLOW** — range of motion is insufficient
- **OUT_OF_RANGE** — joint angle exceeds safe bounds or posture is misaligned

The model is trained on labeled reps collected from the Raspberry Pi itself and deployed back on-device via TensorFlow Lite for fully local inference.

---

## System Pipeline

```
Camera ──► MediaPipe Pose ──► Shoulder/Elbow Keypoints ──┐
                                                          ├──► Feature Fusion
IMU ──────► Orientation Classifier (midterm model) ───────┘
                                                          │
                                              TFLite Form Classifier
                                                          │
                         ┌─────────────────────────────────┤
                         │                                 │
                     LCD Display                   LED / Servo Feedback
               (rep count, grade, cue)           (green=good, red=correct)
                         │
              Optional: LiteLLM + Piper TTS
                 (spoken coaching cues)
```

---

## Project Structure

```
aipi590_final/
│
├── data/
│   ├── raw/                    # Raw IMU and pose keypoint recordings
│   ├── labeled/                # Labeled rep segments (CORRECT / SHALLOW / OUT_OF_RANGE)
│   └── processed/              # Feature-engineered inputs for model training
│
├── models/
│   ├── imu_orientation/        # Midterm model (reused)
│   ├── form_classifier.py      # Training script for TFLite form model
│   └── form_classifier.tflite  # Deployed model (runs on Pi)
│
├── src/
│   ├── data_collection.py      # Simultaneous IMU + pose keypoint logging
│   ├── pose_inference.py       # MediaPipe pose detection wrapper
│   ├── imu_reader.py           # IMU serial/I2C reader
│   ├── feature_fusion.py       # Combines pose and IMU into model input vector
│   ├── classifier.py           # TFLite inference wrapper
│   ├── feedback.py             # LED, servo, and LCD output controller
│   ├── tts_coach.py            # Optional LiteLLM + Piper TTS coaching cues
│   └── main.py                 # Main application loop
│
├── diagrams/
│   ├── system_architecture.png # System data flow diagram
│   └── wiring_diagram.png      # Circuit wiring diagram (Cirkit Designer)
│
├── notebooks/
│   ├── data_exploration.ipynb  # EDA on collected rep data
│   └── model_training.ipynb    # Feature engineering + model training
│
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Key libraries: `mediapipe`, `tflite-runtime`, `smbus2`, `RPi.GPIO`, `RPLCD`, `piper-tts`

### 2. Collect training data

```bash
python src/data_collection.py --label CORRECT --reps 20
python src/data_collection.py --label SHALLOW --reps 20
python src/data_collection.py --label OUT_OF_RANGE --reps 20
```

### 3. Train and export the model

```bash
jupyter notebook notebooks/model_training.ipynb
```

The notebook exports `form_classifier.tflite` to `models/`.

### 4. Run the coaching system

```bash
python src/main.py
```

The LCD will display system status. Begin performing bicep curls in view of the camera with the IMU attached to your wrist.

---

## Data Collection Protocol

Reps are recorded simultaneously from both sensors at ~30Hz. Each session captures a 3-second window per rep phase. Labels are applied manually at collection time via a button press interface (Button A = CORRECT, Button B = SHALLOW, Button C = OUT_OF_RANGE). Raw data is stored as timestamped CSV files with IMU channels and MediaPipe keypoint coordinates (x, y, z, visibility) for relevant joints.

---

## Analysis and Conclusions

*(To be completed after model training and evaluation. Will include: confusion matrix, per-class accuracy, latency measurements on-device, and qualitative observations from live coaching trials.)*

---

## Wiring Diagram

See `diagrams/wiring_diagram.png` — produced using [Cirkit Designer](https://app.cirkitdesigner.com/project).

**Key connections:**
- MPU-6050 IMU → Raspberry Pi via I2C (SDA/SCL)
- USB Camera → Raspberry Pi USB port
- LCD → SPI or GPIO (depending on display model)
- Green LED → GPIO 17 via 330Ω resistor
- Red LED → GPIO 27 via 330Ω resistor
- Optional servo → GPIO 18 (PWM)

---

## Challenges

*(To be updated throughout the project. Known anticipated challenges: synchronization latency between IMU and camera frames; occlusion of elbow keypoint during extreme curl positions; LCD update rate vs. inference loop rate.)*

---

## Next Steps

- Expand to multiple exercises (lateral raise, shoulder press)
- Add a session logging dashboard (InfluxDB + MQTT)
- Collect data from multiple users to generalize the model
- Explore real-time feedback via audio only (no LCD) for less intrusive wearable use

---

## References

- MediaPipe Pose: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
- TensorFlow Lite on Raspberry Pi: https://www.tensorflow.org/lite/guide/python
- MPU-6050 IMU datasheet
- AIPI 590 course materials (Duke University)

---

## Author

AIPI 590 — Individual Final Project
Duke University, Spring 2026
