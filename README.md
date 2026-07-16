# Frontal Gait-Flow Recognition: The F-BioGate Dataset

[![Conference](https://img.shields.io/badge/IJCB-2026-blue)](https://ijcb2026.ieee-biometrics.org)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A robust Multimodal Gait Recognition System early-fusing synchronized RGB-D-IR Optical Flow and Wearable IMU data. Evaluated on the F-BioGate dataset under strict cross-session Closed Set (Identification) and Open Set (Watchlist) protocols using optimized Machine Learning architectures.**

Developed with the F-BioGate dataset created at **VisionLab** (Department of Computer Science, Sapienza University of Rome) and accepted for publication at the *IEEE/IAPR International Joint Conference on Biometrics (IJCB 2026)*.

---

## Overview
This repository contains the official implementation, data processing pipeline, and evaluation protocols for identifying individuals based on unique locomotor patterns from a highly challenging **frontal-view perspective**. By early-fusing **Visual Motion Data** (Depth, RGB, and Infrared Optical Flow at 60 FPS) with **Kinematic Data** (Wearable Xsens MVN IMU Sensors), the system maintains high precision across flat-ground walking and stair traversal tasks.

The pipeline is rigorously evaluated under:
* A strict cross-session **Closed Set** (Identification) protocol to evaluate pure ranking performance.
* A novel ** Open Set** (Watchlist) protocol designed to stress-test threshold calibration and the system's ability to reject unknown impostors without artificial threshold manipulation.

---

## Technical Methodology

The framework handles heterogeneous data sources and applies classical machine learning pipelines to prevent overfitting on constrained biometric cohorts.

### 1. Visual Descriptors (Video)
To capture temporal motion patterns, the vision pipeline extracts features from three synchronized camera streams (Depth, RGB, IR) using two complementary optical flow techniques:
* **GOFI (Gait Optical Flow Image):** Utilizing **Dense Optical Flow** (Farneback algorithm), we accumulate the magnitude of flow vectors over the video sequence, weighted by a binary foreground mask to generate a spatial energy map.
* **Trace Map:** Utilizing **Sparse Optical Flow** (Lucas-Kanade with Shi-Tomasi detection), we track specific anatomical key-points to create a persistent skeletal history of limb trajectories.
* **Preprocessing:** Background is removed using a temporal median filter ($N=20$) after discarding the first 10 frames for sensor stabilization. Concatenating the flattened GOFI and Trace Map across Depth, RGB, and IR yields a **73,728-dimensional** visual feature space ($V_{\text{video}}$).

### 2. Kinematic Features (IMU)
Since raw inertial logs vary in length based on walking speed, we apply **Statistical Feature Extraction** to the 19 Xsens CSV logs per run. For each of the 986 sensor channels, we compute a **5-dimensional descriptor** over time:
* Mean ($\mu$), Standard Deviation ($\sigma$), Minimum ($\min$), Maximum ($\max$), and Root Mean Square ($\text{RMS}$).
Concatenating these descriptors yields a **4,930-dimensional** inertial vector ($V_{\text{imu}}$).

### 3. Early Fusion & Classification Architectures
The normalized visual and inertial vectors are concatenated via Early Fusion ($X_{\text{input}} = [V_{\text{video}} \parallel V_{\text{imu}}]$), yielding a comprehensive **78,658-dimensional** feature space.

* **Closed Set (Identification):** An optimized **Random Forest** (500 trees) operating on the raw fused vector without PCA achieves **94.30% Rank-1 accuracy** (reaching **96.05%** with Trimodal Depth+IR+IMU fusion) and a Macro F1-score of 0.94.
* **Open Set (Watchlist):** To simulate a real-world security checkpoint, the true identity of 19 valid subjects is mathematically masked to generate *Virtual Impostors*, evaluated alongside 7 unseen *True Impostors* (348 total attacks vs 228 genuine attempts). The Random Forest achieves an Equal Error Rate (EER) of **13.99%** and a Detect and Identify Rate (DIR) of **54.82% at a restrictive 1.0% FAR** (**75.88% DIR @ 5.0% FAR**).
* **Deep Learning & SOTA Baselines:** Alternative architectures (1D-CNN, MLP, Siamese Networks, HGB, Linear SVM) were evaluated but exhibited severe probability calibration collapse (overconfidence) in Open Set scenarios, yielding EERs between 78%–95%.

---

## Dataset Structure (F-BioGate)
To ensure reproducibility and zero data leakage, the F-BioGate dataset is organized across distinct temporal sessions (`FirstRun`, `SecondRun`, `ThirdRun`). Models are trained strictly on Runs 1 & 2 and evaluated on Run 3.

---

## Repository Structure

### Root Directories
* `code/`: Contains all Python scripts for data engineering, feature extraction, baseline comparisons, post-hoc analysis, and machine learning.
* `results/`: Contains comprehensive performance evaluations partitioned by architecture (`rf_walk_stairs`, `alt_models`) and analysis type (`CCA`). Includes Ablation Studies, CMC curves, ROC curves, Confidence Histograms, t-SNE manifold plots, and detailed text reports.

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
* `6_closed_set_rf.py`: Handles Grid Search optimization, cross-session training, and CMC curve generation for the primary architecture.
* `6b_closed_set_post_hoc_analysis.py`: Performs qualitative error analysis, generating Confusion Matrices, Confidence Reports, and a comprehensive 15-combination Modality Ablation Study.
* `7_open_set_masked_rf.py`: Executes the rigorous Masked Watchlist protocol, calculating FAR, FRR, DIR, EER, and operational security thresholds, while plotting KDE score distributions.

#### **Post-Hoc Analysis & Visualization**
* `8_cca_video_imu.py`: Performs Canonical Correlation Analysis (CCA) across sessions to investigate out-of-sample visual-inertial relationships and evaluate IMU supervisory signals for future Knowledge Distillation.
* `9_manifold_visualization.py`: Generates publication-ready t-SNE manifold projections comparing Video and IMU feature spaces to illustrate identity preservation and semantic gaps.

#### **Alternative Baselines & Utils**
* `alt_models/`: Contains Closed Set and Masked Open Set evaluation scripts for benchmarking Deep Learning models (1D-CNN, MLP, Siamese) and other classical classifiers (HGB, Linear SVM).
* `utils/`: Contains core computer vision and gait processing logic (`gait_processing.py`).

---

## Getting Started

### Installation
Ensure you are using Python 3.12+. We recommend setting up a clean virtual environment:
```bash
git clone [https://github.com/lorenzomussoo/Frontal-Gait-Flow-Recognition.git](https://github.com/lorenzomussoo/Frontal-Gait-Flow-Recognition.git)
cd Frontal-Gait-Flow-Recognition

# Install required dependencies
pip install numpy opencv-python pandas scikit-learn pyrealsense2 joblib rich matplotlib seaborn torch scipy
