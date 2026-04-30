# Frontal Gait-Flow Recognition 
**Advanced Multimodal Biometric Identification System using Optical Flow, Infrared, and IMU Kinematics.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This project implements a pilot study of a biometric system for identifying individuals based on their unique gait (walking) patterns from a highly challenging **frontal-view perspective**. By early-fusing **Visual Motion Data** (Depth, RGB, and Infrared Optical Flow) with **Kinematic Data** (Wearable IMU Sensors), the system maintains high precision across locomotor tasks, including linear flat-ground walking and stairs traversal. 

The pipeline is rigorously evaluated under a strict cross-session **Closed Set** (Identification) benchmark, and a novel **Masked Open Set** (Watchlist) protocol designed to stress-test the system's threshold calibration and its ability to reject unauthorized impostors.

---

## Technical Methodology

The framework is designed to handle heterogeneous data sources and applies classical machine learning pipelines to prevent overfitting on constrained biometric cohorts.

### 1. Visual Descriptors (Video)
To capture temporal motion patterns, the vision pipeline extracts features from three distinct camera streams (Depth, RGB, IR) using two complementary optical flow techniques:
* **GOFI (Gait Optical Flow Image):** Utilizing **Dense Optical Flow** (Farneback algorithm), we accumulate the magnitude of flow vectors over the gait cycle to generate a spatial energy map.
* **Trace Map:** Utilizing **Sparse Optical Flow** (Lucas-Kanade with Shi-Tomasi detection), we track specific anatomical key-points to create a persistent skeletal history of limb trajectories.
* **Preprocessing:** Background is removed using a temporal median filter (N=20) after discarding the first 10 frames for sensor stabilization.

### 2. Kinematic Features (IMU)
Since raw inertial logs vary in length based on the walking speed, we apply **Statistical Feature Extraction** to the raw CSV logs (Xsens).
For each of the 986 sensor channels, we compute a **5-dimensional descriptor**:
* Mean, Standard Deviation, Minimum, Maximum, and Root Mean Square (RMS).

### 3. Classification Architectures & Protocols
The extracted features (78,658 dimensions) are processed through learning architectures tailored for high-dimensional spaces:

* **Closed Set (Identification):** Utilizes an optimized **Random Forest** algorithm processing the raw fused vector (Depth + RGB + IR + IMU) without dimensionality reduction, achieving **94.08% Rank-1 accuracy** and a Macro F1-score of 0.94.
* **Masked Open Set (Watchlist):** The system transitions to a practical security watchlist. By mathematically masking the true identity of 19 valid subjects to generate *Virtual Impostors*, and introducing 7 completely unseen *True Impostors*, the Random Forest proves its robustness by achieving an Equal Error Rate (EER) of **13.99%** (maintaining a 54.82% detection rate at a strict 1.0% FAR). 
* **Deep Learning Baselines:** Alternative architectures (1D-CNN, MLP, Siamese Networks, HGB, SVM) were comprehensively evaluated but exhibited severe probability calibration collapse (overconfidence) in the Open Set scenario.

---

## Repository Structure

### Root Directories
* `code/`: Contains all Python scripts for data engineering, feature extraction, baseline comparisons, and machine learning.
* `results/`: Contains comprehensive performance evaluations, partitioned by architecture (`rf_walk_stairs` and `alt_models`) and protocol (`closed_set` / `masked_open_set`). Includes Ablation Studies, CMC curves, ROC curves, Confidence Histograms, and detailed text reports.

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
* `5c_data_integrity_check.py`: Applies cryptographic MD5 hashing to guarantee absolute separation (zero leakage) between Train and Test sets.
* `5d_check_and_fix_dataset.py`: Aligns the dataset by safely backing up excess acquisitions to ensure a perfectly balanced protocol.

#### **Machine Learning: Main Pipeline (Random Forest)**
* `6_closed_set_rf.py`: Handles Grid Search optimization, training, and CMC curve generation for the primary architecture.
* `6b_closed_set_post_hoc_analysis.py`: Performs qualitative error analysis, generating Confusion Matrices, Confidence Reports, and a comprehensive 15-combination Modality Ablation Study.
* `7_open_set_masked_rf.py`: Executes the rigorous Masked Watchlist protocol, calculating FAR, FRR, DIR, EER, and operational security thresholds, while plotting KDE score distributions.

#### **Alternative Baselines & Utils**
* `alt_models/`: Contains the Closed Set and Masked Open Set evaluation scripts for benchmarking Deep Learning models (1D-CNN, MLP, Siamese) and other classical models (HGB, SVM).
* `utils/`: Contains core computer vision logic (`gait_processing.py`).

---

## Getting Started

### Installation
Ensure you are using Python 3.12+.
```bash
git clone [https://github.com/lorenzomussoo/Frontal-Gait-Flow-Recognition.git](https://github.com/lorenzomussoo/Frontal-Gait-Flow-Recognition.git)
cd Frontal-Gait-Flow-Recognition
pip install numpy opencv-python pandas scikit-learn pyrealsense2 joblib rich matplotlib seaborn torch scipy
