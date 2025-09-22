# train_hybrid

import os
# disable Apple Metal plugin on macOS if you hit plugin errors
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

# 2) Data augmentation (moderate)
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode="nearest"
)
datagen.fit(x_train)


def build_hybrid():
    m = models.Sequential(name="HybridModel")
    # ─── Float stem ─────────────────────────────────────────
    m.add(layers.Input(shape=(32,32,3), name="input_layer"))
    m.add(layers.Conv2D(32, (3,3), padding="same", activation="relu", name="stem_conv1"))
    m.add(layers.MaxPooling2D((2,2), name="stem_pool1")) 
    m.add(layers.Conv2D(64, (3,3), padding="same", activation="relu", name="stem_conv2"))
    m.add(layers.MaxPooling2D((2,2), name="stem_pool2"))  
    
    # ─── Binarized trunk ─────────────────────────────────────
    for i, filters in enumerate([128, 128, 128], start=1):
        m.add(QuantConv2D(
            filters, (3,3), padding="same", use_bias=False,
            kernel_quantizer="ste_sign",
            kernel_constraint=WeightClip(1.0),
            name=f"bin_conv{i}"
        ))
        m.add(layers.BatchNormalization(name=f"bin_bn{i}"))
        m.add(layers.Activation("relu", name=f"bin_act{i}"))
        m.add(layers.MaxPooling2D((2,2), name=f"bin_pool{i}"))  # halves each time
    
    # ─── Float head ─────────────────────────────────────────
    m.add(layers.Flatten(name="flatten"))
    m.add(layers.Dense(256, activation="relu", name="head_dense1"))
    m.add(layers.Dropout(0.5, name="head_do1"))
    m.add(layers.Dense(10, activation="softmax", name="output_layer"))
    return m

model = build_hybrid()
model.summary()

#  Compile
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 5) Callbacks
checkpoint_cb = ModelCheckpoint(
    "hybrid_best.h5", monitor="val_accuracy",
    save_best_only=True, mode="max", verbose=1
)
earlystop_cb = EarlyStopping(
    monitor="val_accuracy", patience=7,
    restore_best_weights=True, verbose=1
)
reduce_lr_cb = ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.5,
    patience=3, min_lr=1e-6, verbose=1
)

# 6) Train
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=30,
    validation_data=(x_test, y_test),
    callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb],
    verbose=2
)

# 7) Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nHybrid model test accuracy: {acc:.4f}")

# 8) Plot accuracy & loss
plt.figure(figsize=(8,4))
plt.plot(history.history["accuracy"],   label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Hybrid Model Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend(); plt.tight_layout(); plt.savefig("hybrid_accuracy.png"); plt.show()

plt.figure(figsize=(8,4))
plt.plot(history.history["loss"],   label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Hybrid Model Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.tight_layout(); plt.savefig("hybrid_loss.png"); plt.show()

# 9) Save final backup
model.save("hybrid_final.h5")
print("✅ Hybrid model saved to hybrid_final.h5")