'''
AI Traffic Police - Full Pipeline Test
========================================
This script runs the complete pipeline (vehicle counting, emergency
vehicle detection, and traffic violation detection) together, on a set
of real-world street-level traffic images. It depends on the models
and functions already defined in:
  - notebooks/part1_classification_counting.py (yolo_model, count_vehicles)
  - notebooks/part2_emergency_and_violations.py (model, detect_violations_v2) '''

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# ---- Step 1: Locate the sample images ----
sample_dir = '/kaggle/input/datasets/shreyanshnain/testingmodel1'
sample_images = os.listdir(sample_dir)
print(sample_images)


# ---- Step 2: Define the full pipeline function ----
def full_pipeline_test(image_path):
    """
    Runs the complete AI Traffic Police pipeline on one image:
    1. Vehicle counting (YOLO)
    2. Emergency vehicle check (our trained CNN, run on the whole image)
    3. Traffic violation detection (restricted zone + triple-riding)
    """
    # 1) Vehicle counting
    counts, result = count_vehicles(image_path, yolo_model)
    # 2) Emergency vehicle check
    img_for_emergency = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_for_emergency) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    emergency_pred = model.predict(img_array, verbose=0)[0][0]
    is_emergency = emergency_pred > 0.5
    # 3) Violation detection
    img_preview = cv2.imread(image_path)
    h, w = img_preview.shape[:2]
    restricted_zone = [0, int(h * 0.65), int(w * 0.5), h]
    annotated_img, violations = detect_violations_v2(image_path, yolo_model, restricted_zone)
    summary = {
        'vehicle_counts': counts,
        'is_emergency_vehicle': bool(is_emergency),
        'emergency_confidence': f"{emergency_pred:.0%}" if is_emergency else f"{1 - emergency_pred:.0%}",
        'violations': violations if violations else ['None']
    }
    return annotated_img, summary


# ---- Step 3: Run on all sample images and display results ----
plt.figure(figsize=(18, 14))
for i, filename in enumerate(sample_images):
    img_path = os.path.join(sample_dir, filename)
    annotated_img, summary = full_pipeline_test(img_path)
    print(f"\n=== {filename} ===")
    print(f"Vehicle counts: {summary['vehicle_counts']}")
    print(f"Emergency vehicle: {summary['is_emergency_vehicle']} (confidence: {summary['emergency_confidence']})")
    print(f"Violations: {summary['violations']}")
    plt.subplot(2, 3, i + 1)
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    title = f"{filename}\nTotal vehicles: {summary['vehicle_counts']['total']} | Violations: {summary['violations']}"
    plt.title(title, fontsize=8)
plt.tight_layout()
plt.savefig('full_pipeline_test_street.png', dpi=150)
plt.show()
print("\nSaved as full_pipeline_test_street.png")



#-----------LIVE DEMO------------
!pip install gradio -q

import gradio as gr

def gradio_pipeline(uploaded_image):
    # Gradio gives us the image as a numpy array (RGB) — save it so our
    # existing pipeline (which takes a file path) can use it unchanged.
    temp_path = '/kaggle/working/gradio_input.jpg'
    cv2.imwrite(temp_path, cv2.cvtColor(uploaded_image, cv2.COLOR_RGB2BGR))

    annotated_img, summary = full_pipeline_test(temp_path)
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    summary_text = (
        f"Vehicle counts: {summary['vehicle_counts']}\n"
        f"Emergency vehicle: {summary['is_emergency_vehicle']} "
        f"(confidence: {summary['emergency_confidence']})\n"
        f"Violations: {summary['violations']}"
    )
    return annotated_img_rgb, summary_text

demo = gr.Interface(
    fn=gradio_pipeline,
    inputs=gr.Image(label="Upload a traffic image"),
    outputs=[
        gr.Image(label="Annotated Output"),
        gr.Textbox(label="Detection Summary", lines=6)
    ],
    title="AI Traffic Police - Live Demo",
    description="Upload a street/traffic image to run vehicle counting, emergency vehicle detection, and violation detection."
)

demo.launch(share=True)   # share=True gives a public link you can even open on your phone
