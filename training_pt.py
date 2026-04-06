"""
training_pt.py
--------------
Trains a Random Forest on pt_movement_data.csv and saves the model package.

Same step-through pipeline as training_arm.py from the midterm.
Feature set is expanded from 6 to 10:
  pitch + roll + gyro for each of 3 segments (9 IMU features)
  + fused_angle from fusion.py (1 feature)

The extra confusion matrix step (option 6) is worth running before saving.
With 7 movement state classes — some of which are the same physical movement
at different speeds or ranges — the model can confuse adjacent states.
The matrix shows exactly where that happens so you can decide whether to
collect more data for specific labels before deploying.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


# ----------------------------
# Configuration
# ----------------------------

CSV_FILE   = "pt_movement_data.csv"
MODEL_FILE = "model_package_pt.joblib"

SEGMENTS = [
    "shoulder",
    "upper_arm",
    "forearm",
]

# 9 IMU features + fused_angle — must match CSV_HEADER in data_collection_pt.py
FEATURE_COLS = [
    f"{seg}_{val}"
    for seg in SEGMENTS
    for val in ("pitch", "roll", "gyro")
] + ["fused_angle"]

N_ESTIMATORS = 100
RANDOM_STATE = 42
TEST_SIZE    = 0.2


# ----------------------------
# Global state — same pattern as midterm
# ----------------------------

df      = None
X_train = None
X_test  = None
y_train = None
y_test  = None
model   = None
encoder = None


# ----------------------------
# Pipeline steps
# ----------------------------

def load_data():
    """Load CSV and drop rows with any missing feature values."""
    global df
    df = pd.read_csv(CSV_FILE)
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS)
    print(f"Loaded {before} rows. {len(df)} remain after dropping incomplete reads.")
    print(f"\nLabel counts:\n{df['label'].value_counts()}\n")


def encode_labels():
    """Convert string labels to integers. Save encoder for inference."""
    global df, encoder
    encoder = LabelEncoder()
    df["label_encoded"] = encoder.fit_transform(df["label"])
    mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print(f"Label encoding: {mapping}\n")


def split_data():
    """Split into train/test sets."""
    global X_train, X_test, y_train, y_test
    X = df[FEATURE_COLS]
    y = df["label_encoded"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows\n")


def train_model():
    """Train the Random Forest."""
    global model
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    print(f"Trained Random Forest with {N_ESTIMATORS} trees.\n")


def evaluate_model():
    """Print accuracy and per-label breakdown."""
    preds    = model.predict(X_test)
    accuracy = (preds == y_test).mean()
    print(f"Overall accuracy: {accuracy:.3f}\n")
    print("Per-label breakdown:")
    print(classification_report(y_test, preds, target_names=encoder.classes_))


def print_confusion_matrix():
    """
    Print the confusion matrix as a labeled table.

    Rows are true labels, columns are predicted labels.
    Off-diagonal cells show where the model gets confused.

    For movement states, the most likely confusions are:
      ascending_good <-> too_fast   (same motion, different speed)
      too_shallow    <-> peak_good  (similar arm position, different range)
    If either of those cells is high, collect more data for those labels
    before deploying.
    """
    preds  = model.predict(X_test)
    labels = encoder.classes_
    cm     = confusion_matrix(y_test, preds)
    cm_df  = pd.DataFrame(cm, index=labels, columns=labels)

    print("Confusion matrix (rows = true, cols = predicted):\n")
    print(cm_df.to_string())
    print()


def print_feature_importances():
    """
    Show which features the model relies on most.

    fused_angle ranking is worth noting — if it's near the top,
    the MediaPipe + IMU fusion is genuinely contributing. If it's
    at the bottom, the raw IMU features are doing most of the work.
    """
    ranked = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    print("Feature importances (all 10):")
    for feat, score in ranked:
        bar = "#" * int(score * 50)
        print(f"  {feat:<22}: {score:.4f}  {bar}")
    print()


def save_model():
    """Bundle model, feature columns, and encoder into one file."""
    package = {
        "model":        model,
        "feature_cols": FEATURE_COLS,
        "encoder":      encoder,
    }
    joblib.dump(package, MODEL_FILE)
    print(f"Model package saved to {MODEL_FILE}\n")


# ----------------------------
# Menu
# ----------------------------

def print_menu():
    print("\nMenu:")
    print(f"  1) Load {CSV_FILE}")
    print("  2) Encode labels")
    print("  3) Train/test split")
    print("  4) Train Random Forest")
    print("  5) Evaluate model")
    print("  6) Print confusion matrix")
    print("  7) Print feature importances")
    print(f"  8) Save model package ({MODEL_FILE})")
    print("  9) Quit")


def main():
    print("PT Movement — Model Training\n")

    while True:
        print_menu()
        choice = input("\nChoose (1-9): ").strip()

        if choice == "1":
            try:
                load_data()
            except FileNotFoundError:
                print(f"ERROR: {CSV_FILE} not found. Collect data first.\n")

        elif choice == "2":
            if df is None:
                print("Load data first (option 1).")
                continue
            encode_labels()

        elif choice == "3":
            if encoder is None:
                print("Encode labels first (option 2).")
                continue
            split_data()

        elif choice == "4":
            if X_train is None:
                print("Split data first (option 3).")
                continue
            train_model()

        elif choice == "5":
            if model is None:
                print("Train model first (option 4).")
                continue
            evaluate_model()

        elif choice == "6":
            if model is None:
                print("Train model first (option 4).")
                continue
            print_confusion_matrix()

        elif choice == "7":
            if model is None:
                print("Train model first (option 4).")
                continue
            print_feature_importances()

        elif choice == "8":
            if model is None:
                print("Train model first (option 4).")
                continue
            save_model()

        elif choice == "9":
            print("Done.")
            break

        else:
            print("Invalid choice. Pick 1-9.")


if __name__ == "__main__":
    main()