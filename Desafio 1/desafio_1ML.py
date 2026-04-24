import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ─── 1. EXTRAIR DATASET ───────────────────────────────────────────────────────
ZIP_PATH = "kagglecatsanddogs_5340.zip"
EXTRACT_DIR = "PetImages"

if not os.path.exists(EXTRACT_DIR):
    print("Extraindo dataset...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(".")
    print("Extração concluída.")
else:
    print("Dataset já extraído.")

# ─── 2. LIMPAR IMAGENS CORROMPIDAS ────────────────────────────────────────────
print("\nVerificando imagens corrompidas...")
removed = 0
for category in ["Cat", "Dog"]:
    folder = os.path.join(EXTRACT_DIR, category)
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        try:
            img = tf.io.read_file(fpath)
            tf.image.decode_jpeg(img)
        except Exception:
            os.remove(fpath)
            removed += 1
print(f"Imagens removidas: {removed}")

# ─── 3. PARÂMETROS ────────────────────────────────────────────────────────────
IMG_SIZE   = (160, 160)
BATCH_SIZE = 32
EPOCHS_FROZEN   = 10
EPOCHS_FINETUNE = 10
VALIDATION_SPLIT = 0.2

# ─── 4. DATA GENERATORS COM AUGMENTATION ─────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=VALIDATION_SPLIT,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=VALIDATION_SPLIT,
)

train_gen = train_datagen.flow_from_directory(
    EXTRACT_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    seed=42,
)

val_gen = val_datagen.flow_from_directory(
    EXTRACT_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    seed=42,
)

print(f"\nClasses: {train_gen.class_indices}")
print(f"Imagens treino : {train_gen.samples}")
print(f"Imagens validação: {val_gen.samples}")

# ─── 5. CONSTRUIR MODELO COM TRANSFER LEARNING ───────────────────────────────
base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False   # congela o backbone

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# ─── 6. FASE 1 — TREINAR APENAS O TOPO ───────────────────────────────────────
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy"),
    ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_accuracy"),
]

print("\n=== Fase 1: backbone congelado ===")
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FROZEN,
    callbacks=callbacks,
)

# ─── 7. FASE 2 — FINE-TUNING (descongelar últimas 30 camadas) ────────────────
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"])

print("\n=== Fase 2: fine-tuning ===")
history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINETUNE,
    callbacks=callbacks,
)

# ─── 8. GRÁFICOS ─────────────────────────────────────────────────────────────
def plot_history(h1, h2):
    acc  = h1.history["accuracy"]      + h2.history["accuracy"]
    val  = h1.history["val_accuracy"]  + h2.history["val_accuracy"]
    loss = h1.history["loss"]          + h2.history["loss"]
    vloss= h1.history["val_loss"]      + h2.history["val_loss"]
    ep   = range(1, len(acc) + 1)
    sep  = len(h1.history["accuracy"]) # época onde começa o fine-tuning

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(ep, acc,  label="Train acc")
    axes[0].plot(ep, val,  label="Val acc")
    axes[0].axvline(sep, color="gray", linestyle="--", label="Fine-tune start")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(ep, loss,  label="Train loss")
    axes[1].plot(ep, vloss, label="Val loss")
    axes[1].axvline(sep, color="gray", linestyle="--", label="Fine-tune start")
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("Gráfico salvo em training_curves.png")

plot_history(history1, history2)

# ─── 9. AVALIAÇÃO FINAL ───────────────────────────────────────────────────────
loss, acc = model.evaluate(val_gen)
print(f"\nAcurácia final na validação: {acc:.4f}")

# ─── 10. PREDIÇÃO EM AMOSTRAS ─────────────────────────────────────────────────
class_names = {v: k for k, v in train_gen.class_indices.items()}
images, labels = next(val_gen)
preds = model.predict(images[:9])

plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])
    pred_class = class_names[int(preds[i] > 0.5)]
    true_class = class_names[int(labels[i])]
    color = "green" if pred_class == true_class else "red"
    plt.title(f"Pred: {pred_class}\nReal: {true_class}", color=color)
    plt.axis("off")
plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=150)
plt.show()
print("Predições salvas em sample_predictions.png")

print("\nModelo salvo em best_model.keras")