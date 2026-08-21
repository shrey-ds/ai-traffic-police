# AI Traffic Police 

A computer vision system that helps traffic authorities monitor roads, detect incidents, and improve traffic management — built for Parts 1 (Foundations) and 2 (Detection) of the Computer Vision track.

## What this project does
                   Capability                            Approach
Classify vehicle types (car, bus, motorcycle, truck) :	CNN trained from scratch

Count vehicles in an image	                         :  YOLOv8

Detect emergency vehicles	                           :  CNN trained from scratch

Detect basic traffic violations	                     : Rule-based logic on top of YOLOv8 detections


                                  My Approach and reasoning

### Part 1a — Vehicle Classification (CNN from scratch)

I trained a CNN from scratch (no pretrained weights) on the compulsory dataset provided in track pdf. 

---> Key design decisions:

- The dataset is **multi-label**, not single-class which is a single photo can contain both a car and a bus. I treated this as a multi-label problem (`sigmoid` output + `binary_crossentropy` loss) rather than a single "pick one" classifier (`softmax`), which would be the wrong tool here.
- The compulsory dataset's "motorcycle" label is used to represent the "bike" category from the brief, since that's the label the dataset actually provides.
- Test accuracy: 60.34%. This is a genuinely harder task than a typical single-label image classifier, since the model has to independently judge the presence of 4 vehicle types per image.
  

### Part 1b — Vehicle Counting (pretrained YOLOv8)

Counting requires knowing *where* each vehicle is in an image (bounding boxes), but the compulsory Part 1 dataset only provides whole-image labels ("this image contains a car and a bus"), with no location information. Rather than force that dataset to do a job it isn't built for, we used a **pretrained YOLOv8 model** (explicitly permitted for this track) to detect and count vehicles directly, no additional training needed, since YOLO already recognizes car/bus/truck/motorcycle/bicycle from its COCO training.

### Part 2a — Emergency Vehicle Detection (CNN from scratch)

I sourced a dedicated **"Emergency vs Non-Emergency Vehicle"** dataset (from Kaggle) and trained a second CNN from scratch to classify a vehicle image as emergency or not.

---> Key design decisions: 
- Added data augmentation (rotation, zoom, flip, brightness) since this dataset is much smaller than Part 1's — this helps a from-scratch CNN generalize rather than memorize.
- Used `BatchNormalization` and `GlobalAveragePooling2D` instead of a large `Dense` layer, to reduce overfitting risk on a smaller dataset.
- Used early stopping (`patience=5` on validation loss) so training stops automatically once it's no longer improving.
- **Test accuracy: 70.45%**, with high-confidence correct predictions on close-up test images (ambulance: 80%, fire truck: 98%).

However, a known limitation: this classifier looks at the *whole image*, not individual detected vehicles. It performs well when the emergency vehicle is the clear subject of the photo, but confidence drops when it's smaller or partially visible within a busier street scene (see `part3_full_pipeline_test.py` results — a fire truck in a busy scene was correctly detected by YOLO but scored close to the 50% decision boundary by the emergency classifier).

### Part 2b — Traffic Violation Detection (rule-based, on YOLO detections)

I implemented two violation types on top of YOLO's detections:

1. Restricted-zone violation: flags any vehicle whose bounding box overlaps a marked zone (e.g. a no-parking area or pedestrian crossing). This is the same core technique real traffic-camera systems use ("virtual tripwire" / zone detectors). In a real deployment, this zone would be calibrated once to a fixed camera's actual restricted area; for this demo, I use a placeholder zone position since my test images come from varied, uncalibrated angles.
2. Triple-riding violation: flags a motorcycle with 3 or more `person` detections overlapping it, a common and genuinely enforced traffic violation in many countries. This reuses YOLO's existing `person` and `motorcycle` classes with no extra training required.

I deliberately avoided implementing a "red-light" violation, since that would require live video with signal-state information, which is out of scope for a still-image pipeline and as I didn't want to fake a violation type I couldn't honestly support.

## Results

- `training_curves.png` / `emergency_classifier_training_curves.png` — accuracy/loss over training epochs for both CNNs
- `sample_predictions.png` / `emergency_sample_predictions.png` — real test images with true vs. predicted labels
- `vehicle_counting_proof.png` — YOLO detection + counting on test images
- `violation_detection_proof.png` — restricted-zone and triple-riding detection on test images
- `full_pipeline_test_street.png` — all three systems (counting, emergency detection, violations) run together on real street-level traffic photos

## Setup

```bash
pip install -r requirements.txt
```

Run the scripts in order inside a Kaggle notebook (or any environment with GPU access and the datasets attached):
1. `part1_classification_counting.py`
2. `part2_emergency_and_violations.py`
3. `part3_full_pipeline_test.py`

### Datasets I used
- **Part 1 (compulsory):** provided vehicle multiclass datasets (Vehicles-coco.v2i.multiclass, Vehicles.v1i.multiclass)
- **Part 2a:** [Emergency vs Non-Emergency Vehicle Classification](https://www.kaggle.com/datasets/abhisheksinghblr/emergency-vehicles-identification) (Kaggle)
- **Part 2b test images:** a small set of street-level traffic photos (Pexels, free to use): not used for training, only to demonstrate the violation-detection pipeline

### A note on "using YOLO"

I used YOLOv8 in its pretrained form (no additional training) for vehicle counting and as the detection backbone for violation logic. I interpreted "you may use YOLO" as permission to use it as a ready tool, consistent with explicit confirmation I received that pretrained YOLO use is allowed for this track. Both of my required from-scratch models: the vehicle classifier (Part 1a) and the emergency vehicle classifier (Part 2a), were trained entirely from scratch on labeled data, and do not use any pretrained weights. 

## Honest limitations in what i designed

- Vehicle classifier accuracy (~60%) and emergency classifier accuracy (~70%) are modest, since both were trained from scratch in limited time on comparatively small/medium datasets. .
- YOLOv8-nano (the fastest, smallest YOLO variant) struggles with small or distant vehicles in aerial/top-down camera angles — I found street-level camera angles work significantly better, and adjusted my test images accordingly.
- The restricted-zone violation uses a placeholder zone position rather than a calibrated real camera zone, since my test images aren't from a single fixed camera feed.
- The emergency vehicle classifier runs on whole images rather than per-vehicle crops, which limits its reliability in busy multi-vehicle scenes (see Part 2a section above for detail).

## Tools I used

TensorFlow/Keras, OpenCV, Ultralytics YOLOv8, pandas, scikit-learn, matplotlib
