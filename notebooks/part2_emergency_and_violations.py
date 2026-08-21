import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from ultralytics import YOLO


# ============================================================
# PART 2A: EMERGENCY VEHICLE CLASSIFICATION (CNN from scratch)
# ============================================================

# ---- Step 1: Load labels ----
# Update this path once you've added the Kaggle dataset to your notebook's
# Input panel (Add Data -> search "Emergency vs Non-Emergency Vehicle").

DATA_DIR = '/kaggle/input/datasets/abhisheksinghblr/emergency-vehicles-identification/Emergency_Vehicles'
CSV_PATH = os.path.join(DATA_DIR, 'train.csv')
IMG_DIR = os.path.join(DATA_DIR, 'train')

df = pd.read_csv(CSV_PATH)
df['filepath'] = df['image_names'].apply(lambda x: os.path.join(IMG_DIR, x))
df['emergency_or_not'] = df['emergency_or_not'].astype(str)  # flow_from_dataframe wants string class labels

print("Total images:", len(df))
print(df['emergency_or_not'].value_counts())   # sanity check: how balanced are the two classes?

# ---- Step 2: Split into train / validation / test ----
# stratify=... keeps the emergency/non-emergency ratio the same in every split
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['emergency_or_not'], random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['emergency_or_not'], random_state=42)

print("Train:", train_df.shape, "Valid:", valid_df.shape, "Test:", test_df.shape)

# ---- Step 3: Image generators (mirrors Part 1's setup) ----
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# A bit of augmentation this time, since this dataset is much smaller than
# Part 1's - augmentation helps the from-scratch CNN generalize better and
# should also help push accuracy above Part 1's 58%.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)
eval_datagen = ImageDataGenerator(rescale=1./255)   # no augmentation for validation/test - we want a true read

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df, x_col='filepath', y_col='emergency_or_not',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)
valid_gen = eval_datagen.flow_from_dataframe(
    dataframe=valid_df, x_col='filepath', y_col='emergency_or_not',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)
test_gen = eval_datagen.flow_from_dataframe(
    dataframe=test_df, x_col='filepath', y_col='emergency_or_not',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False
)

# Step 4: Build the CNN (from scratch, same family as Part 1) ----
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),          # stabilizes training - helps since dataset is smaller
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.GlobalAveragePooling2D(),      # fewer params than Flatten+Dense, less overfitting risk
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    # Single output: probability that this is an emergency vehicle.
    # Sigmoid + binary_crossentropy because this is a plain two-class problem
    # (unlike Part 1, which was multi-label).
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---- Step 5: Train, with early stopping so we don't overfit a small dataset ----
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=30,
    callbacks=[early_stop]
)

# ---- Step 6: Evaluate ----
test_loss, test_accuracy = model.evaluate(test_gen)
print(f"Emergency Vehicle Classifier - Test Accuracy: {test_accuracy:.2%}")
print(f"Emergency Vehicle Classifier - Test Loss: {test_loss:.4f}")

# ---- Step 7: Training curves ----
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Emergency Classifier - Accuracy over Epochs')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Emergency Classifier - Loss over Epochs')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

plt.tight_layout()
plt.savefig('emergency_classifier_training_curves.png')
plt.show()

# ---- Step 8: Save the model ----
model.save('emergency_vehicle_classifier.keras')
print("Model saved as emergency_vehicle_classifier.keras")

# ---- Step 9: Sample predictions ----
test_images, test_labels = next(iter(test_gen))
predictions = model.predict(test_images)

plt.figure(figsize=(15, 8))
for i in range(min(6, len(test_images))):
    plt.subplot(2, 3, i + 1)
    plt.imshow(test_images[i])
    plt.axis('off')
    true_label = "Emergency" if test_labels[i] == 1 else "Non-Emergency"
    pred_label = "Emergency" if predictions[i][0] > 0.5 else "Non-Emergency"
    confidence = predictions[i][0] if predictions[i][0] > 0.5 else 1 - predictions[i][0]
    plt.title(f"True: {true_label}\nPred: {pred_label} ({confidence:.0%})", fontsize=9)
plt.tight_layout()
plt.savefig('emergency_sample_predictions.png')
plt.show()


# ============================================================
# PART 2B: TRAFFIC VIOLATION DETECTION (rule-based, on YOLO detections)
# ============================================================

yolo_model = YOLO('yolov8n.pt')
vehicle_classes = ['car', 'bus', 'truck', 'motorcycle', 'bicycle']


def check_box_overlap(box1, box2):
    """
    Checks if two bounding boxes overlap at all.
    box format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return x2 > x1 and y2 > y1


def count_vehicles(image_path, model, conf_threshold=0.3):
    """
    Runs YOLO detection on a single image and returns a dictionary
    with the count of each vehicle type found, plus a total count.
    (Reused from Part 1b.)
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


def detect_violations_v2(image_path, model, restricted_zone, conf_threshold=0.3):
    """
    Detects two types of traffic violations:

    1. Restricted-zone violation: any vehicle whose bounding box overlaps
       a marked restricted zone (e.g. pedestrian crossing, no-parking zone,
       no-entry lane).

    2. Triple-riding violation: a motorcycle with 3+ people overlapping it.

    restricted_zone: [x1, y1, x2, y2] - the marked zone's coordinates
    """
    results = model(image_path, conf=conf_threshold, verbose=False)
    result = results[0]
    img = cv2.imread(image_path)

    violations = []
    vehicle_boxes = []
    person_boxes = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if class_name in vehicle_classes:
            vehicle_boxes.append((class_name, [x1, y1, x2, y2]))
        elif class_name == 'person':
            person_boxes.append([x1, y1, x2, y2])
    for class_name, (x1, y1, x2, y2) in vehicle_boxes:
        is_zone_violation = check_box_overlap([x1, y1, x2, y2], restricted_zone)
        is_tripleriding_violation = False
        if class_name == 'motorcycle':
            riders = sum(1 for p_box in person_boxes if check_box_overlap([x1, y1, x2, y2], p_box))
            is_tripleriding_violation = riders >= 3
        has_violation = is_zone_violation or is_tripleriding_violation
        color = (0, 0, 255) if has_violation else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        label_parts = [class_name]
        if is_zone_violation:
            label_parts.append("RESTRICTED ZONE")
            violations.append(f"{class_name}: entered restricted zone")
        if is_tripleriding_violation:
            label_parts.append("TRIPLE-RIDING")
            violations.append(f"{class_name}: triple-riding violation")
        cv2.putText(img, " | ".join(label_parts), (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # Draw the restricted zone as a semi-transparent rectangle
    overlay = img.copy()
    cv2.rectangle(overlay, (restricted_zone[0], restricted_zone[1]),
                   (restricted_zone[2], restricted_zone[3]), (0, 165, 255), -1)
    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
    cv2.rectangle(img, (restricted_zone[0], restricted_zone[1]),
                   (restricted_zone[2], restricted_zone[3]), (0, 165, 255), 2)
    return img, violations


# ---- Test on 6 images for stronger visual proof ----
plt.figure(figsize=(18, 10))
for i in range(6):
    img_path = test_df['filepath'].iloc[i]

    img_preview = cv2.imread(img_path)
    h, w = img_preview.shape[:2]

    # Define a restricted zone: bottom-left corner area of the frame
    restricted_zone = [0, int(h * 0.65), int(w * 0.5), h]

    annotated_img, violations = detect_violations_v2(img_path, yolo_model, restricted_zone)

    plt.subplot(2, 3, i + 1)
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Violations: {violations if violations else 'None'}", fontsize=9)
plt.tight_layout()
plt.savefig('violation_detection_proof.png', dpi=150)
plt.show()
print("Saved as violation_detection_proof.png")

# Note: the full end-to-end pipeline test (counting + emergency check +violations run together on real street images) lives in a separate script: part3_full_pipeline_test.py
