# train_cnn

import os
# Disable Apple Metal plugin for macOS
os.environ["TF_METAL_DISABLE"] = "1"

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets, utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)
import matplotlib.pyplot as plt

# Load & preprocess CIFAR-10
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test .astype("float32") / 255.0
y_train = utils.to_categorical(y_train, 10)
y_test  = utils.to_categorical(y_test, 10)


def build_cnn():
    m = models.Sequential([
        layers.Input(shape=(32,32,3), name="input_layer"),
        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D((2,2)),  # 32→16

        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D((2,2)),  # 16→8

        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D((2,2)),  # 8→4

        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax", name="output_layer"),
    ])
    return m

model = build_cnn()
model.summary()

#  Compile with an initial LR of 1e-3
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
checkpoint_cb = ModelCheckpoint(
    filepath="cnn_baseline_updated.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)
earlystop_cb = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
    verbose=1
)
reduce_lr_cb = ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# 5) Train
history = model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.2,
    callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb],
    verbose=2
)

# 6) Final evaluation
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nFinal test accuracy: {test_acc:.4f}")

# 7) Print per‐epoch metrics
print("\nEpoch metrics:")
for i, (ta, va) in enumerate(
    zip(history.history["accuracy"], history.history["val_accuracy"]), 1):
    print(f"  Epoch {i:2d}: Train={ta:.3f}, Val={va:.3f}")

# 8) Plot & save accuracy curve
plt.figure(figsize=(8,4))
plt.plot(history.history["accuracy"],   label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("CNN Accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend(); plt.tight_layout()
plt.savefig("cnn_accuracy_curve.png")
plt.show()

# 9) Plot & save loss curve
plt.figure(figsize=(8,4))
plt.plot(history.history["loss"],   label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("CNN Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.tight_layout()
plt.savefig("cnn_loss_curve.png")
plt.show()

# 10) Backup save of final epoch
model.save("cnn_baseline_final.h5")
print("✅ Final CNN saved to cnn_baseline_final.h5")