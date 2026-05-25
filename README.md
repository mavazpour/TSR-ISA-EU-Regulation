# Traffic Sign Recognition and EU Regulations on Intelligent Speed Assistance

This repository contains the official implementation of a real-time, two-stage Traffic Sign Recognition (TSR) pipeline engineered to align with the safety mandates and latency constraints governed by European Union regulations.

The project evaluates the performance of the streamlined YOLO26m architecture under adverse environmental conditions, validating its deployment feasibility on resource-constrained embedded automotive hardware.

---

## Technical Architecture Overview

The perception system is divided into two distinct sequential vision stages to fulfill the real-world speed limit determination window mandated by the regulations:

1. Stage 1: Localization (GTSDB): Processes full-resolution driving environment frames (1280 x 1280 pixels) using a modified YOLO26m architecture to extract precise spatial coordinates and isolate regions of interest (RoIs) containing traffic signs.
2. Stage 2: Classification (GTSRB): Ingests the localized, cropped RoIs (640 x 640 pixels) into a secondary network specialized in determining precise semantic speed limit categories.

---

## Experimental Results

The models were trained and evaluated utilizing the Google Colab Pro platform equipped with an NVIDIA A100-SXM4-40GB GPU. The quantitative metrics captured from the best validation checkpoints are structured below:

| Pipeline Stage / Dataset | Precision (P) | Recall (R) | mAP50 |
| :--- | :---: | :---: | :---: |
| Stage 1: Localization (GTSDB) | 0.888 | 0.914 | 0.970 |
| Stage 2: Classification (GTSRB) | 0.942 | 0.921 | 0.950 |

---

## Repository Structure

```
├── assets/                  # Diagrams, HUD visual assets, and performance plots
├── src/
│   ├── preprocessing.py     # Adaptive Histogram Equalization and spatial normalization
│   ├── train_localization.py# YOLO26m localization training script (GTSDB configuration)
│   ├── train_classification.py# YOLO26m semantic classification training script (GTSRB configuration)
│   ├── inference_pipeline.py# sequential execution pipeline and temporal verification
│   └── driver_hud.py        # Real-time Head-Up Display telemetry interface
├── requirements.txt         # Dependencies and execution environment boundaries
└── README.md                # Project documentation
```

---

## Key Features Implemented

* NMS-Free Prediction Head: Integrates the direct bounding box regression architecture of YOLO26 to eliminate Non-Maximum Suppression (NMS) latency overhead.
* Temporal Verification Algorithm: Requires a candidate traffic sign to be localized across multiple consecutive frames before updating the driver interface, actively mitigating transient false positives.
* Dynamic Region of Interest (ROI) Mapping: Restricts initial frame processing geometrically to regions where roadside signs are predictably positioned.
* Interactive Telemetry HUD: Displays current speed constraints dynamically alongside a tracking bar showing historical detections and sequence counts.

---

## Core Dependencies

To replicate the execution environment, ensure the following core libraries are installed:

* Python >= 3.8
* PyTorch >= 2.0
* Ultralytics (configured for YOLO26 architectures)
* OpenCV-Python
* NumPy

Install all package boundaries via:
```
pip install -r requirements.txt
```

---

## Academic Context

This study was conducted as part of the seminar Grundlagen der Fahrerassistenz- und der aktiven Sicherheitssysteme at Technische Hochschule Ingolstadt (THI) during the Summer Semester 2026.

* Author: Mohammadali Avazpour
* Academic Supervisor: Professor Alexander Knorr
* Institution: Faculty of Electrical Engineering and Information Technology, Technische Hochschule Ingolstadt (THI)
