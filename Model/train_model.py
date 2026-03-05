import os
import sys
import ast
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, confusion_matrix,
                             ConfusionMatrixDisplay)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# ── STEP 1: Load dataset ─────────────────────────────────────────
# ast.literal_eval converts the string "[120, 84, 91...]" back to an actual list
# when we saved to CSV, arrays became strings... this converts them back

def load_dataset(csv_path):
    print("[INFO] Loading dataset...")
    df = pd.read_csv(csv_path)

    df["flow"]   = df["flow"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    df["thorac"] = df["thorac"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    df["spo2"]   = df["spo2"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

    # drop classes with too few samples for SMOTE to work
    # Body event (3 samples) and Mixed Apnea (2 samples) are too rare to learn from
    df = df[~df["label"].isin(["Body event", "Mixed Apnea"])]
    df = df.reset_index(drop=True)

    print(f"  Total windows: {len(df)}")
    print(f"  Label counts:\n{df['label'].value_counts()}")
    return df


# ── STEP 2: Build features ───────────────────────────────────────
# We join flow + thorac + spo2 into one long array per window
# Shape of one sample: (2040,) = flow(960) + thorac(960) + spo2(120)

def build_features(df):
    X = []
    for _, row in df.iterrows():
        combined = np.concatenate([row["flow"], row["thorac"], row["spo2"]])
        X.append(combined)

    # X shape: (num_windows, 2040)
    X = np.array(X)

    # CNN needs shape (num_windows, 2040, 1)
    # the extra 1 = number of channels (like RGB has 3, we have 1)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X


# ── STEP 3: Build CNN model ──────────────────────────────────────
# Simple 1D CNN:
#   Conv1D  → slides a filter across the time series, detects local patterns
#   MaxPool → shrinks the output, keeps the strongest signal
#   Flatten → converts 2D output to 1D list
#   Dense   → fully connected layer, makes final decision

def build_model(input_length, num_classes):
    model = models.Sequential([

        # first conv layer: 32 filters, each looks at 5 timesteps at a time
        layers.Conv1D(filters=32, kernel_size=5,
                      activation="relu", input_shape=(input_length, 1)),
        layers.MaxPooling1D(pool_size=2),

        # second conv layer: 64 filters, learns deeper patterns
        layers.Conv1D(filters=64, kernel_size=5, activation="relu"),
        layers.MaxPooling1D(pool_size=2),

        # flatten and classify
        layers.Flatten(),
        layers.Dense(64, activation="relu"),

        # output layer: one node per class, softmax gives probabilities
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",           # standard optimizer
        loss="sparse_categorical_crossentropy",  # for integer labels
        metrics=["accuracy"]
    )
    return model


# ── STEP 4: LOPO Cross Validation ───────────────────────────────
# Leave One Participant Out:
#   each participant gets a turn as the test set
#   model trains on the other 4, tests on this one

def run_lopo(df, X, y, label_encoder):
    participants = df["participant"].unique()
    num_classes  = len(label_encoder.classes_)

    # store results across all folds
    all_true  = []
    all_pred  = []

    for test_participant in participants:
        print(f"\n[FOLD] Test participant: {test_participant}")

        # boolean mask: True where participant matches
        test_mask  = df["participant"] == test_participant
        train_mask = ~test_mask   # ~ means NOT → everything else

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test  = X[test_mask]
        y_test  = y[test_mask]

        print(f"  Train size: {len(X_train)}  Test size: {len(X_test)}")

        # SMOTE needs 2D input → flatten from (n, 2040, 1) to (n, 2040)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)

        # apply SMOTE on training data only
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_flat, y_train)
        print(f"  After SMOTE train size: {len(X_train_bal)}")

        # reshape back to (n, 2040, 1) for CNN
        X_train_bal = X_train_bal.reshape(X_train_bal.shape[0], X_train_bal.shape[1], 1)

        # build a fresh model for each fold
        model = build_model(input_length=X.shape[1], num_classes=num_classes)

        # train the model
        # epochs=10 means the model sees the full training data 10 times
        # batch_size=32 means it updates weights after every 32 samples
        # verbose=0 means no progress bar printed
        model.fit(X_train_bal, y_train_bal, epochs=3, batch_size=32, verbose=0)

        # predict on test participant
        y_pred_probs = model.predict(X_test, verbose=0)
        # argmax picks the class with highest probability
        y_pred = np.argmax(y_pred_probs, axis=1)

        all_true.extend(y_test)
        all_pred.extend(y_pred)

        # per fold metrics
        acc = accuracy_score(y_test, y_pred)
        print(f"  Fold Accuracy: {acc:.3f}")

    return np.array(all_true), np.array(all_pred)


# ── STEP 5: Report metrics ───────────────────────────────────────

def report_metrics(all_true, all_pred, label_encoder, out_dir):
    class_names = label_encoder.classes_

    print("\n════════════════════════════════")
    print("        FINAL RESULTS")
    print("════════════════════════════════")

    acc = accuracy_score(all_true, all_pred)
    print(f"Accuracy  : {acc:.3f}")

    # weighted averages account for class imbalance
    # (we have way more Normal windows than event windows)
    prec = precision_score(all_true, all_pred, average="weighted", zero_division=0)
    rec  = recall_score(all_true, all_pred,    average="weighted", zero_division=0)
    print(f"Precision : {prec:.3f}")
    print(f"Recall    : {rec:.3f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_true, all_pred)
    print(cm)
    print(f"Classes: {class_names}")

    # save confusion matrix as image
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45)
    plt.title("Confusion Matrix — LOPO")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(out_path)
    plt.close()
    print(f"\nConfusion matrix saved → {out_path}")


# ── STEP 6: Main ─────────────────────────────────────────────────

def main():
    csv_path = os.path.join("Dataset", "breathing_dataset.csv")
    out_dir  = os.path.join("Results")

    # load
    df = load_dataset(csv_path)

    # build features
    X = build_features(df)

    # encode labels: "Normal"→0, "Hypopnea"→1, "Obstructive Apnea"→2 etc
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])
    print(f"\n  Classes: {label_encoder.classes_}")

    # run LOPO
    all_true, all_pred = run_lopo(df, X, y, label_encoder)

    # report
    report_metrics(all_true, all_pred, label_encoder, out_dir)


if __name__ == "__main__":
    main()