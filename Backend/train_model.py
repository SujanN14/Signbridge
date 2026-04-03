"""
ISL Two-Hand Gesture — MLP Training Pipeline
=============================================
Production-grade training script for the SignBridge gesture classifier.
Trains a regularized Multi-Layer Perceptron on the 126-dim landmark CSV
produced by test_camera.py (extract_landmarks).

Outputs
-------
- isl_twohand_mlp.h5   : Best Keras model
- label_encoder.pkl    : Fitted sklearn LabelEncoder for inference
- feature_scaler.pkl   : Fitted StandardScaler for inference
- training_report.txt  : Full classification report + confusion matrix
- training_curves.png  : Loss & accuracy plots for visual diagnostics
"""

import os
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import LabelEncoder, StandardScaler
from sklearn.metrics            import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models        import Sequential
from tensorflow.keras.layers        import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils         import to_categorical
from tensorflow.keras.callbacks     import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers  import l2
from tensorflow.keras.optimizers    import Adam

warnings.filterwarnings("ignore")

# ─────────────────────────── Configuration ────────────────────────────────── #

CSV_PATH        = "isl_twohand_landmarks.csv"
MODEL_OUT       = "isl_twohand_mlp.h5"
ENCODER_OUT     = "label_encoder.pkl"
SCALER_OUT      = "feature_scaler.pkl"
REPORT_OUT      = "training_report.txt"
CURVES_OUT      = "training_curves.png"
CHECKPOINT_PATH = "checkpoints/best_model.h5"

HIDDEN_UNITS    = [512, 256, 128]
DROPOUT_RATE    = 0.35
L2_LAMBDA       = 1e-4
LEARNING_RATE   = 1e-3
BATCH_SIZE      = 32
MAX_EPOCHS      = 150
PATIENCE        = 12
LR_PATIENCE     = 6

TEST_SIZE       = 0.20
VAL_SIZE        = 0.20
RANDOM_STATE    = 42

# ──────────────────────────── Logging Setup ───────────────────────────────── #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────── Reproducibility ──────────────────────────────── #

os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ─────────────────────────── Data Loading ─────────────────────────────────── #

def load_dataset():
    if not Path(CSV_PATH).exists():
        raise FileNotFoundError(
            f"Dataset not found at '{CSV_PATH}'. Run test_camera.py first."
        )

    df = pd.read_csv(CSV_PATH)
    logger.info(f"Dataset loaded  |  Shape: {df.shape}")

    if "label" not in df.columns:
        raise ValueError("CSV is missing the 'label' column.")
    if df.isnull().values.any():
        null_count = df.isnull().sum().sum()
        logger.warning(f"Found {null_count} NaN values — filling with 0.")
        df.fillna(0.0, inplace=True)

    X = df.drop("label", axis=1).values.astype(np.float32)
    y = df["label"].values

    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"Classes ({len(unique)}): { dict(zip(unique, counts)) }")
    min_samples = counts.min()
    if min_samples < 10:
        logger.warning(
            f"Class '{unique[counts.argmin()]}' has only {min_samples} samples."
        )

    encoder       = LabelEncoder()
    y_encoded     = encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    logger.info(f"Feature dim     |  {X.shape[1]}")
    logger.info(f"Num classes     |  {y_categorical.shape[1]}")

    return X, y_categorical, encoder, y_encoded

# ──────────────────────────── Preprocessing ───────────────────────────────── #

def preprocess(X, y_categorical, y_encoded):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical,
        test_size=TEST_SIZE,
        stratify=y_encoded,
        random_state=RANDOM_STATE,
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    logger.info(f"Split  →  Train: {len(X_train)}  |  Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, scaler

# ────────────────────────── Class Weights ─────────────────────────────────── #

def get_class_weights(y_train):
    y_indices = np.argmax(y_train, axis=1)
    classes   = np.unique(y_indices)
    weights   = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_indices
    )
    class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))
    logger.info(f"Class weights computed for {len(classes)} classes.")
    return class_weight_dict

# ──────────────────────────── Model ───────────────────────────────────────── #

def build_model(input_dim, num_classes):
    model = Sequential(name="SignBridge_MLP")

    for i, units in enumerate(HIDDEN_UNITS):
        if i == 0:
            model.add(Dense(units, activation="relu",
                            input_shape=(input_dim,),
                            kernel_regularizer=l2(L2_LAMBDA),
                            name=f"dense_{i+1}"))
        else:
            model.add(Dense(units, activation="relu",
                            kernel_regularizer=l2(L2_LAMBDA),
                            name=f"dense_{i+1}"))
        model.add(BatchNormalization(name=f"bn_{i+1}"))
        model.add(Dropout(DROPOUT_RATE, name=f"dropout_{i+1}"))

    model.add(Dense(num_classes, activation="softmax", name="output"))
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=logger.info)
    return model

# ──────────────────────────── Callbacks ───────────────────────────────────── #

def build_callbacks():
    Path(CHECKPOINT_PATH).parent.mkdir(parents=True, exist_ok=True)
    early_stop = EarlyStopping(
        monitor="val_loss", patience=PATIENCE,
        restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=LR_PATIENCE, min_lr=1e-6, verbose=1
    )
    checkpoint = ModelCheckpoint(
        filepath=CHECKPOINT_PATH, monitor="val_accuracy",
        save_best_only=True, verbose=0
    )
    return [early_stop, reduce_lr, checkpoint]

# ──────────────────────────── Training ────────────────────────────────────── #

def train_model(model, X_train, y_train, class_weight_dict):
    logger.info("Training started ...")
    history = model.fit(
        X_train, y_train,
        validation_split=VAL_SIZE,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=build_callbacks(),
        class_weight=class_weight_dict,
        verbose=1,
    )
    logger.info(
        f"Training finished  |  "
        f"Best val_accuracy: {max(history.history['val_accuracy']):.4f}  |  "
        f"Stopped at epoch {len(history.history['loss'])}"
    )
    return history

# ──────────────────────────── Evaluation ──────────────────────────────────── #

def evaluate_model(model, X_test, y_test, encoder):
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test Loss       |  {loss:.4f}")
    logger.info(f"Test Accuracy   |  {acc:.4f}")

    y_pred      = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true      = np.argmax(y_test, axis=1)
    class_names = encoder.classes_.tolist()

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm     = confusion_matrix(y_true, y_pred)

    report_text = (
        f"SignBridge ISL — MLP Evaluation Report\n"
        f"{'='*50}\n"
        f"Test Loss     : {loss:.4f}\n"
        f"Test Accuracy : {acc:.4f}\n\n"
        f"Classification Report\n{'-'*50}\n{report}\n\n"
        f"Confusion Matrix\n{'-'*50}\n{cm}\n"
    )
    Path(REPORT_OUT).write_text(report_text)
    logger.info(f"Report saved    |  '{REPORT_OUT}'")
    print(f"\n{report_text}")

# ────────────────────────── Training Curves ───────────────────────────────── #

def save_training_curves(history):
    epochs = range(1, len(history.history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "SignBridge MLP — Training Curves",
        fontsize=15, fontweight="bold", y=1.02,
    )

    # ── Left subplot: Loss ────────────────────────────────────────────────────
    axes[0].plot(
        epochs, history.history["loss"],
        color="red", linewidth=2, label="Training Loss",
    )
    axes[0].plot(
        epochs, history.history["val_loss"],
        color="orange", linewidth=2, linestyle="--", label="Validation Loss",
    )
    axes[0].set_title("Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Categorical Crossentropy", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # ── Right subplot: Accuracy ───────────────────────────────────────────────
    axes[1].plot(
        epochs, history.history["accuracy"],
        color="blue", linewidth=2, label="Training Accuracy",
    )
    axes[1].plot(
        epochs, history.history["val_accuracy"],
        color="green", linewidth=2, linestyle="--", label="Validation Accuracy",
    )
    axes[1].set_title("Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy", fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(CURVES_OUT, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Curves saved    |  '{CURVES_OUT}'")

# ────────────────────────── Save Artifacts ────────────────────────────────── #

def save_artifacts(model, encoder, scaler):
    model.save(MODEL_OUT)
    logger.info(f"Model saved     |  '{MODEL_OUT}'")
    joblib.dump(encoder, ENCODER_OUT)
    logger.info(f"Encoder saved   |  '{ENCODER_OUT}'")
    joblib.dump(scaler, SCALER_OUT)
    logger.info(f"Scaler saved    |  '{SCALER_OUT}'")

# ──────────────────────────── Entry Point ─────────────────────────────────── #

def main():
    logger.info("=" * 60)
    logger.info("  SignBridge — MLP Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"TensorFlow      |  {tf.__version__}")
    logger.info(f"GPU available   |  {bool(tf.config.list_physical_devices('GPU'))}")

    X, y_categorical, encoder, y_encoded = load_dataset()
    X_train, X_test, y_train, y_test, scaler = preprocess(X, y_categorical, y_encoded)
    class_weight_dict = get_class_weights(y_train)
    model   = build_model(input_dim=X_train.shape[1], num_classes=y_categorical.shape[1])
    history = train_model(model, X_train, y_train, class_weight_dict)
    evaluate_model(model, X_test, y_test, encoder)
    save_training_curves(history)   # ← called after model.fit() completes
    save_artifacts(model, encoder, scaler)

    logger.info("=" * 60)
    logger.info("  Pipeline complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()