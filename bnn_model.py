# train_bnn

import os
# Disable Apple Metal plugin for macOS
os.environ["TF_METAL_DISABLE"] = "1"

import numpy as np
import tensorflow as tf
import larq as lq
from larq.layers import QuantConv2D, QuantDense
from larq.constraints import WeightClip
from tensorflow.keras import layers, models, datasets, utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 1) Load & preprocess CIFAR-10
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test .astype("float32") / 255.0
y_train = utils.to_categorical(y_train, 10)
y_test  = utils.to_categorical(y_test, 10)

# 2) Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,      
    width_shift_range=0.1,  
    height_shift_range=0.1, 
    horizontal_flip=True,
    zoom_range=0.1          
    
)
datagen.fit(x_train)


def build_bnn():
    m = models.Sequential()
    m.add(layers.Input((32,32,3), name="input_layer"))

    # First layer stays full precision (standard practice)
    m.add(layers.Conv2D(32, (3,3), padding="same", use_bias=False))
    m.add(layers.BatchNormalization())
    m.add(layers.Activation("relu"))
    m.add(layers.MaxPooling2D((2,2)))

    
    m.add(QuantConv2D(
        32, (3,3),
        padding="same",
        use_bias=False,
        kernel_quantizer="ste_sign",
        kernel_constraint=WeightClip(1)
    ))
    m.add(layers.BatchNormalization())
    m.add(layers.Activation("relu"))
    m.add(layers.MaxPooling2D((2,2)))

    # Another binary layer
    m.add(QuantConv2D(
        64, (3,3),
        padding="same",
        use_bias=False,
        kernel_quantizer="ste_sign",
        kernel_constraint=WeightClip(1)
    ))
    m.add(layers.BatchNormalization())
    m.add(layers.Activation("relu"))
    m.add(layers.MaxPooling2D((2,2)))

    m.add(layers.Flatten())
    
    # dense layers for speed
    m.add(layers.Dense(64, activation="relu"))  
    m.add(layers.Dropout(0.3))
    
    m.add(layers.Dense(32, activation="relu"))  
    m.add(layers.Dropout(0.2))

    m.add(layers.Dense(10, activation="softmax", name="output_layer"))
    return m

model = build_bnn()

# 4) Compile with adjusted LR for smaller model
model.compile(
    optimizer=Adam(learning_rate=1e-3),  # Slightly higher LR for smaller model
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# 5) Callbacks: best‐val checkpoint, early stopping, LR reduction
checkpoint_cb = ModelCheckpoint(
    filepath="bnn_baseline_updated.h5",
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

# 6) Train with adjusted batch size
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=128),  
    epochs=25,    
    validation_data=(x_test, y_test),
    callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb],
    verbose=2
)

# 7) Final evaluation
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nFinal test accuracy: {acc:.4f}")

# 8) Print per-epoch metrics
print("\nEpoch metrics:")
for i, (ta, va) in enumerate(
    zip(history.history["accuracy"], history.history["val_accuracy"]), 1):
    print(f"  Epoch {i:2d}: Train={ta:.3f}, Val={va:.3f}")

# 9) Plot and save curves
plt.figure(figsize=(8,4))
plt.plot(history.history["accuracy"],   label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("BNN Accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend(); plt.tight_layout()
plt.savefig("bnn_accuracy_curve.png")
plt.show()

plt.figure(figsize=(8,4))
plt.plot(history.history["loss"],   label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("BNN Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.tight_layout()
plt.savefig("bnn_loss_curve.png")
plt.show()

# 10) Backup save of the final epoch
model.save("bnn_baseline_final1.h5")
print("✅ Final BNN saved to bnn_baseline_final1.h5")
