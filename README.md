# Frontal Gait-Flow Recognition
**Advanced Multimodal Biometric Identification System using Optical Flow, Infrared, and IMU Kinematics.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This project implements a state-of-the-art biometric system that identifies individuals by their unique gait (walking) patterns. By fusing **Visual Motion Data** (Depth, RGB, and Infrared Optical Flow) with **Kinematic Data** (Wearable IMU Sensors), the system maintains high precision across various actions, including flat ground walking, stairs, and slopes. 

The pipeline is rigorously evaluated under both **Closed Set** (Identification) and **Open Set** (Watchlist/Authentication) protocols to ensure real-world forensic applicability.

---

## Technical Methodology

The framework is designed to handle heterogeneous data sources and applies specialized machine learning pipelines based on the available modalities.

### 1. Visual Descriptors (Video)
*Applicable to **Walk** and **Stairs** tasks.*
To capture temporal motion patterns, the vision pipeline extracts features from three distinct camera streams (Depth, RGB, IR) using two complementary optical flow techniques:
* **GOFI (Gait Optical Flow Image):** Utilizing **Dense Optical Flow** (Farneback algorithm), we accumulate the magnitude of flow vectors over the gait cycle to generate a spatial energy map.
* **Trace Map:** Utilizing **Sparse Optical Flow** (Lucas-Kanade with Shi-Tomasi detection), we track specific anatomical key-points to create a skeletal history of limb trajectories.
* **Preprocessing:** Background is removed using a temporal median filter (N=20) after discarding the first 10 frames for sensor stabilization.

### 2. Kinematic Features (IMU)
*Applicable to **All Tasks** (Sole modality for **Slope**).*
Since raw inertial logs vary in length based on the walking speed, we apply **Statistical Feature Extraction** to the raw CSV logs (Xsens).
For each sensor channel, we compute a **5-dimensional descriptor**:
* Mean, Standard Deviation, Minimum, Maximum, and Root Mean Square (RMS).

### 3. Classification Architectures & Protocols
The extracted features are processed through learning architectures tailored to the specific task:

* **Walk/Stairs (Multimodal):** Utilizes an optimized **Random Forest** algorithm. It directly processes the massive 78,658-dimensional fused vector (Depth + RGB + IR + IMU) without dimensionality reduction, achieving **94.08% Rank-1 accuracy** in Closed Set identification.
* **Slope (IMU-Only):** Utilizes a **Linear Support Vector Machine (SVM)**. The 4,930-dimensional inertial vector is first compressed via PCA (retaining 95% variance), achieving **81.58% Rank-1 accuracy**.

**Open Set Watchlist:** The system transitions from a theoretical classifier to a practical security watchlist, proving its capability to reject unknown impostors with an Equal Error Rate (EER) of **15.35% (RF)** and **21.16% (SVM)** under strict 15-known subject evaluations.

---

## Repository Structure

### Root Directories
* `code/`: Contains all Python scripts for data engineering, feature extraction, baseline comparisons, and machine learning.
* `models/`: Stores serialized `.joblib` models and optimal hyperparameters `.json` for both Closed and Open Set configurations.
* `results/`: Contains comprehensive performance evaluations, partitioned by task (`rf_walk_stairs` and `svm_slope`) and protocol (`closed_set` / `open_set`). Includes Ablation Studies, CMC curves, ROC curves, Confidence Histograms, and detailed text reports.

### Source Code (`code/`)
The codebase is structured sequentially to replicate the entire research pipeline:

#### **Data Engineering & Video Processing**
* `1_explore_bag.py`: ROS bag inspection and stream visualization.
* `2_convert_bags.py` & `2b_fix_subjects.py`: Extracts RGB/Depth/IR streams and resolves subject naming conflicts.
* `2c_audit_video.py`: Scans and validates video health, detecting corrupted or incomplete frames.

#### **IMU Organization & Dataset Auditing**
* `3_organize_imu.py` & `3b_audit_imu.py`: Parses raw ZIP archives, standardizes inertial data, and audits sequence availability.
* `4_verify_dataset_completeness.py`: Enforces strict dataset completeness across Video and IMU modalities for all sessions.

#### **Feature Extraction & Integrity Checks**
* `5_multimodal_feature_extractor.py`: The core engine fusing GOFI, Trace Maps, and IMU data into unified `.npy` feature vectors.
* `5b_check_dims.py`: Verifies the structural integrity and dimensionality (78,658 for multimodal) of the processed vectors.
* `5c_data_integrity_check.py`: Applies cryptographic hashing to guarantee absolute separation (zero leakage) between Train and Test sets.
* `5d_check_and_fix_dataset.py`: Aligns the dataset by safely backing up excess acquisitions to ensure a perfectly balanced protocol.

#### **Machine Learning: Core Pipeline**
* `6a_closed_set_rf_train_walk_stairs.py` & `6b_closed_set_svm_train_slope.py`: Handles Grid Search optimization, training, and CMC curve generation for the main proposed architectures.
* `6c_closed_set_post_hoc_analysis.py`: Performs qualitative error analysis, generating Confusion Matrices, Confidence Reports, and a 15-combination Modality Ablation Study.
* `7a_open_set_rf_walk_stairs.py` & `7b_open_set_svm_slope.py`: Executes the Watchlist protocol, calculating FAR, FRR, DIR, EER, and operational security thresholds.
* `7c_open_set_plot_distributions.py`: Generates Kernel Density Estimation (KDE) distributions to visualize genuine vs. impostor score overlap.

#### **Alternative Models & Utils**
* `alt_models/`: Contains evaluation scripts for benchmark models and the aggregated `scores.txt` log used for paper comparisons.
* `utils/`: Contains core computer vision logic (`gait_processing.py`).

---

## Getting Started

### Installation
Ensure you are using Python 3.12+.
```bash
git clone [https://github.com/lorenzomussoo/Frontal-Gait-Flow-Recognition.git](https://github.com/lorenzomussoo/Frontal-Gait-Flow-Recognition.git)
cd Frontal-Gait-Flow-Recognition
pip install numpy opencv-python pandas scikit-learn pyrealsense2 joblib rich matplotlib seaborn torch
