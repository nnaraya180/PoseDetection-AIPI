"""
training_arm.py
---------------
1. Loads arm_position_data.csv
2. Trains a Random Forest classifier on pitch/roll from 3 arm IMUs
3. Saves a model package (model, feature columns, label encoder)

6 features total: pitch + roll for shoulder, upper_arm, forearm.
Same step-through pipeline as all previous training files.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


# ----------------------------
# Configuration
# ----------------------------

CSV_FILE   = "arm_position_data.csv"
MODEL_FILE = "model_package_arm.joblib"

# Must match SEGMENTS order in data_collection_arm.py
SEGMENTS = [
    "shoulder",
    "upper_arm",
    "forearm",
]

FEATURE_COLS = [
    f"{seg}_{angle}"
    for seg in SEGMENTS
    for angle in ("pitch", "roll")
]

N_ESTIMATORS = 100
RANDOM_STATE = 42
TEST_SIZE    = 0.2


# ----------------------------
# Global state
# Same pattern as all previous training files and class labs
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
    """Load CSV and drop rows with any missing sensor values."""
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


def print_feature_importances():
    """
    Show which sensors and angles the model relies on most.
    This is a great diagnostic — if shoulder_pitch dominates,
    the model is mainly using shoulder elevation to classify.
    If forearm_roll matters a lot, the elbow bend is key.
    """
    ranked = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    print("Feature importances (all 6):")
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
    print("  6) Print feature importances")
    print(f"  7) Save model package ({MODEL_FILE})")
    print("  8) Quit")


def main():
    print("Arm Position — Model Training\n")

    while True:
        print_menu()
        choice = input("\nChoose (1-8): ").strip()

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
            print_feature_importances()

        elif choice == "7":
            if model is None:
                print("Train model first (option 4).")
                continue
            save_model()

        elif choice == "8":
            print("Done.")
            break

        else:
            print("Invalid choice. Pick 1-8.")


if __name__ == "__main__":
    main()