"""
AI Traffic Police - Part 1: Vehicle Classification & Counting
================================================================
This script has two main parts:
1. CLASSIFICATION - A CNN trained from scratch to identify vehicle types
   (car, bus, motorcycle, truck) in an image. Since one image can contain
   multiple vehicle types, this is treated as a multi-label problem.
2. COUNTING - Uses a pretrained YOLOv8 model to detect and count vehicles
   in an image by drawing bounding boxes around each one.

Dataset: Combined from two Roboflow "multiclass" vehicle datasets
(provided compulsory dataset for Part 1).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models


# ============================================================
# PART 1A: VEHICLE CLASSIFICATION (CNN trained from scratch)
# ============================================================

# ---- Step 1: Load and clean the dataset labels ----
def load_and_clean(split_name):
    """
    Loads the label CSVs for one data split (train/valid/test) from both
    source datasets, standardizes column names, and combines them into
    a single table with image file paths and vehicle-type labels.
    """
    p1 = f'/kaggle/input/datasets/shreyanshnain/vehicles/Vehicles-coco.v2i.multiclass/{split_name}'
    p2 = f'/kaggle/input/datasets/shreyanshnain/vehicles/Vehicles.v1i.multiclass/{split_name}'

    d1 = pd.read_csv(os.path.join(p1, '_classes.csv'))
    d2 = pd.read_csv(os.path.join(p2, '_classes.csv'))

    # Remove accidental extra spaces in column names
    d1.columns = d1.columns.str.strip()
    d2.columns = d2.columns.str.strip()

    # Standardize column names between the two datasets (e.g. Bus -> bus)
    d2 = d2.rename(columns={'Bus': 'bus', 'Motorcycle': 'motorcycle'})
    if '0' in d2.columns:
        d2 = d2.drop(columns=['0'])  # '0' = background/no-vehicle column, not needed

    # Build full file paths for each image
    d1['filepath'] = d1['filename'].apply(lambda x: os.path.join(p1, x))
    d2['filepath'] = d2['filename'].apply(lambda x: os.path.join(p2, x))

    # Keep only the columns we need, in the same order
    d1 = d1[['filepath', 'bus', 'car', 'motorcycle', 'truck']]
    d2 = d2[['filepath', 'bus', 'car', 'motorcycle', 'truck']]

    # Combine both datasets into one table
    combined = pd.concat([d1, d2], ignore_index=True)
    return combined


# Load all three splits (train/validation/test)
train_df = load_and_clean('train')
valid_df = load_and_clean('valid')
test_df = load_and_clean('test')

print("Train:", train_df.shape)
print("Valid:", valid_df.shape)
print("Test:", test_df.shape)

# ---- Step 2: Prepare image data generators ----
# Vehicle types we're predicting. Note: "motorcycle" is used to represent
# the "bike" category, since that's the label the dataset provides.
class_columns = ['bus', 'car', 'motorcycle', 'truck']

# Labels must be numeric (float) for multi-label training
train_df[class_columns] = train_df[class_columns].astype('float32')
valid_df[class_columns] = valid_df[class_columns].astype('float32')
test_df[class_columns] = test_df[class_columns].astype('float32')

IMG_SIZE = (224, 224)   # all images resized to this for consistency
BATCH_SIZE = 32         # number of images processed at once

# Rescale pixel values from 0-255 to 0-1, which helps the model learn better
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_dataframe(
    dataframe=train_df, x_col='filepath', y_col=class_columns,
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='raw'   # 'raw' because an image can have multiple vehicle types at once
)

valid_gen = datagen.flow_from_dataframe(
    dataframe=valid_df, x_col='filepath', y_col=class_columns,
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='raw'
)

test_gen = datagen.flow_from_dataframe(
    dataframe=test_df, x_col='filepath', y_col=class_columns,
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='raw', shuffle=False   # no shuffle, so results stay in a known order
)

# ---- Step 3: Build the CNN model (from scratch, no pretrained weights) ----
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),             # 224x224 color images (R,G,B)

    layers.Conv2D(32, (3, 3), activation='relu'),   # detect simple patterns (edges, colors)
    layers.MaxPooling2D(2, 2),                      # shrink image, keep key info

    layers.Conv2D(64, (3, 3), activation='relu'),   # detect more complex patterns
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),  # detect shapes (wheels, windows, etc.)
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),                     # turn the 2D feature grid into a flat list
    layers.Dense(128, activation='relu'), # combine everything learned so far
    layers.Dropout(0.5),                  # randomly drop connections to reduce overfitting

    # Output: one score (0-1) per vehicle type. Sigmoid (not softmax) is used
    # because each vehicle type is an independent yes/no decision, not a
    # single "pick one" choice.
    layers.Dense(4, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',   # correct loss function for multi-label problems
    metrics=['accuracy']
)

model.summary()

# ---- Step 4: Train the model ----
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=5
)

# ---- Step 5: Evaluate on the test set (final, unseen data) ----
test_loss, test_accuracy = model.evaluate(test_gen)
print(f"Test Accuracy: {test_accuracy:.2%}")
print(f"Test Loss: {test_loss:.4f}")

# ---- Step 6: Plot training curves (accuracy & loss over epochs) ----
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

# ---- Step 7: Save the trained model ----
model.save('vehicle_classifier.keras')
print("Model saved as vehicle_classifier.keras")

# ---- Step 8: Show sample predictions (true label vs. predicted label) ----
test_images, test_labels = next(iter(test_gen))
predictions = model.predict(test_images)

plt.figure(figsize=(15, 8))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(test_images[i])
    plt.axis('off')

    true_classes = [class_columns[j] for j in range(4) if test_labels[i][j] == 1]
    pred_classes = [class_columns[j] for j in range(4) if predictions[i][j] > 0.5]

    plt.title(f"True: {true_classes}\nPred: {pred_classes}", fontsize=9)

plt.tight_layout()
plt.savefig('sample_predictions.png')
plt.show()


# ============================================================
# PART 1B: VEHICLE COUNTING (pretrained YOLOv8)
# ============================================================
# Note: pretrained model use was explicitly permitted for this part.
# YOLOv8 already knows how to detect common vehicle types (car, bus,
# truck, motorcycle, bicycle) since it was trained on the COCO dataset,
# so no additional training is needed for counting.

from ultralytics import YOLO

# Load the pretrained YOLOv8 model (nano version = fastest, lightweight)
yolo_model = YOLO('yolov8n.pt')

# Vehicle classes we care about (from YOLO's built-in COCO classes)
vehicle_classes = ['car', 'bus', 'truck', 'motorcycle', 'bicycle']


def count_vehicles(image_path, model=yolo_model, conf_threshold=0.3):
    """
    Runs YOLO detection on a single image and returns a dictionary
    with the count of each vehicle type found, plus a total count.
    """
    results = model(image_path, conf=conf_threshold, verbose=False)
    result = results[0]

    counts = {v: 0 for v in vehicle_classes}
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        if class_name in vehicle_classes:
            counts[class_name] += 1

    counts['total'] = sum(v for k, v in counts.items() if k != 'total')
    return counts, result


# ---- Quick test: count vehicles in 5 sample test images ----
for i in range(5):
    img_path = test_df['filepath'].iloc[i]
    counts, result = count_vehicles(img_path)
    print(f"Image {i + 1}: {counts}")

# ---- Visual proof: show the 5 images with detected boxes and counts ----
plt.figure(figsize=(18, 10))
for i in range(5):
    img_path = test_df['filepath'].iloc[i]
    counts, result = count_vehicles(img_path)

    # result.plot() draws all detected boxes directly onto the image
    annotated_img = result.plot()

    plt.subplot(2, 3, i + 1)
    # YOLO images are in BGR color order; matplotlib expects RGB, so we flip it
    plt.imshow(annotated_img[..., ::-1])
    plt.axis('off')
    plt.title(f"Vehicle count: {counts['total']}", fontsize=11)

plt.tight_layout()
plt.savefig('vehicle_counting_proof.png', dpi=150)
plt.show()

print("Saved as vehicle_counting_proof.png")
