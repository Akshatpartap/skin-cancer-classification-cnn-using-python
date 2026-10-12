# Skin Cancer Classification Using Convolutional Neural Networks (CNN) and Python

An end-to-end, academic deep learning project developed to classify dermatoscopic skin lesion images into 7 diagnostic categories using a Convolutional Neural Network (CNN) built with TensorFlow/Keras and deployed as an interactive Streamlit web application.

---

## 📌 Table of Contents
1. [Project Overview](#project-overview)
2. [Key Objectives](#key-objectives)
3. [Features](#features)
4. [Technology Stack](#technology-stack)
5. [Dataset Information & Download Guide](#dataset-information--download-guide)
6. [Machine Learning Engineering Highlights](#machine-learning-engineering-highlights)
   - [Preventing Data Leakage via Lesion-Level Split](#preventing-data-leakage-via-lesion-level-split)
   - [Addressing Severe Class Imbalance](#addressing-severe-class-imbalance)
7. [Repository Structure](#repository-structure)
8. [Installation & Windows Setup Guide](#installation--windows-setup-guide)
9. [Step-by-Step Execution Guide](#step-by-step-execution-guide)
   - [Step 1: Download & Prepare the Dataset](#step-1-download--prepare-the-dataset)
   - [Step 2: Train the CNN Model](#step-2-train-the-cnn-model)
   - [Step 3: Evaluate on Held-Out Test Set](#step-3-evaluate-on-held-out-test-set)
   - [Step 4: Predict on a Single Image (CLI)](#step-4-predict-on-a-single-image-cli)
   - [Step 5: Launch the Streamlit Web Application](#step-5-launch-the-streamlit-web-application)
10. [Evaluation Metrics & Integrity Policy](#evaluation-metrics--integrity-policy)
11. [Adding Genuine Screenshots](#adding-genuine-screenshots)
12. [Project Limitations](#project-limitations)
13. [Medical & Ethical Disclaimer](#medical--ethical-disclaimer)
14. [Future Improvements](#future-improvements)
15. [GitHub & LinkedIn Sharing Guide](#github--linkedin-sharing-guide)
16. [Author & Attribution](#author--attribution)
17. [License](#license)

---

## 📖 Project Overview

Skin cancer is among the most diagnosed malignancies worldwide. Early detection of malignant forms, especially **Melanoma**, dramatically increases 5-year survival rates. Dermatoscopy provides magnified, illuminated imaging of skin lesions, but visual interpretation requires extensive clinical expertise.

This academic project implements an introductory computer vision solution:
- Builds and trains a custom 4-block Convolutional Neural Network (CNN) from scratch in Python using TensorFlow and Keras.
- Operates on the benchmark **HAM10000** ("Human Against Machine with 10000 training images") dermatoscopic dataset.
- Packages the trained model into an intuitive, browser-based web dashboard using Streamlit.
- Prioritizes scientific integrity: prevents data leakage through patient/lesion grouping, accounts for class imbalance, and includes prominent clinical disclaimers.

---

## 🎯 Key Objectives

- **Deep Learning Fundamentals**: Build an understandable CNN architecture explaining Convolution, Batch Normalization, Max Pooling, Dropout, and Dense layers.
- **Data Integrity**: Eliminate data leakage by splitting at the `lesion_id` level instead of naive image-level splitting.
- **Robustness**: Compensate for extreme class imbalance (67% common moles vs 1.1% rare lesions) using balanced class weights.
- **Practical Deployment**: Deliver an interactive Streamlit UI for user image uploads, real-time prediction, and full-class confidence breakdown.
- **CPU-Friendly Execution**: Optimize tensor sizes and pipelines so first-year students can run and demonstrate the project on a standard Windows laptop without an external GPU.
- **Honest Academic Reporting**: Eliminate fabricated accuracy claims and clearly separate experimental outputs from example placeholders.

---

## ✨ Features

- **End-to-End Modular Pipeline**: Cleanly separated modules for dataset parsing, CNN architecture, utilities, training, evaluation, inference, and UI.
- **Fault-Tolerant Execution**: Scripts and the Streamlit app never crash if the dataset or trained model is missing; they display clear instructional banners explaining next steps.
- **7-Class Lesion Classification**: Distinguishes between Melanoma, Basal Cell Carcinoma, Actinic Keratoses, Benign Keratoses, Nevi, Dermatofibroma, and Vascular Lesions.
- **Ranked Confidence Output**: Displays top classification along with full confidence distributions across all 7 categories.
- **Comprehensive Visual Reports**: Automatically generates training curves (loss and accuracy over epochs) and normalized confusion matrix heatmaps.

---

## 💻 Technology Stack

| Layer | Library / Tool | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10 / 3.11 | Core programming language |
| **Deep Learning** | TensorFlow 2.15+ / Keras | CNN layer definition, backpropagation, and training |
| **Data Processing** | NumPy & Pandas | Numerical tensors, metadata grouping, and CSV handling |
| **Machine Learning**| Scikit-Learn | Stratified patient splitting, class weights, evaluation metrics |
| **Image Handling** | Pillow (PIL) | Safe image decoding, resizing, and color mode verification |
| **Visualization** | Matplotlib & Seaborn | Confusion matrix heatmaps and training curves |
| **Web Interface** | Streamlit | Browser-based interactive user interface |
| **Version Control**| Git & GitHub | Source tracking and project portfolio hosting |

---

## 📊 Dataset Information & Download Guide

This project is designed to use the **HAM10000** dataset:
- **Title**: *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions*
- **Authors**: Philipp Tschandl, Cliff Rosendahl, Harald Kittler (*Nature Scientific Data*, 2018)
- **License**: Creative Commons Attribution-NonCommercial 4.0 International ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/))
- **Total Samples**: 10,015 dermatoscopic images across 7 classes

### Where to Download:
1. **Kaggle Mirror (Fastest & Simplest)**:
   - [Kaggle: Skin Cancer MNIST (HAM10000)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
2. **Harvard Dataverse (Official Archive)**:
   - [Harvard Dataverse DOI: 10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T)

### Folder Setup:
Extract the downloaded files into your project's `data/raw/` directory:
```text
skin-cancer-classification-cnn/
└── data/
    └── raw/
        ├── HAM10000_metadata.csv
        ├── HAM10000_images_part_1/
        │   ├── ISIC_0024306.jpg
        │   └── ...
        └── HAM10000_images_part_2/
            ├── ISIC_0029306.jpg
            └── ...
```

---

## 🔬 Machine Learning Engineering Highlights

### Preventing Data Leakage via Lesion-Level Split
In HAM10000, multiple photos often exist for the **same lesion** on a patient.
- **The Trap**: If images are randomly split, photo A of a lesion enters the training set while photo B of the exact same lesion enters the test set. The model memorizes that lesion, giving a falsely high test score.
- **Our Solution**: In [`src/dataset.py`](src/dataset.py), we group images by `lesion_id` and perform a stratified split on unique lesions. All images of any given lesion belong strictly to the Training (70%), Validation (15%), or Test (15%) set.

### Addressing Severe Class Imbalance
The 10,015 images are distributed unevenly:
- `nv` (Common moles): 6,705 (67.0%)
- `mel` (Melanoma - malignant): 1,113 (11.1%)
- `bkl` (Benign keratosis): 1,099 (11.0%)
- `bcc` (Basal cell carcinoma - malignant): 514 (5.1%)
- `akiec` (Actinic keratoses - pre-malignant): 327 (3.3%)
- `vasc` (Vascular lesions): 142 (1.4%)
- `df` (Dermatofibroma): 115 (1.1%)

To prevent the model from ignoring the minority malignant classes:
1. We compute **balanced class weights** via Scikit-Learn and pass them to Keras `model.fit(class_weight=...)`.
2. We evaluate with **Macro-averaged Precision, Recall, and F1-Score**, not just overall accuracy.

---

## 📂 Repository Structure

```text
skin-cancer-classification-cnn/
├── .gitignore                     # Excludes virtual environments, datasets, and large weights
├── LICENSE                        # MIT License + CC BY-NC 4.0 data attribution notice
├── README.md                      # Comprehensive academic documentation
├── requirements.txt               # Pinned, tested dependencies for Windows CPU
├── app.py                         # Interactive Streamlit web application
├── train.py                       # CNN training script with early stopping & checkpoints
├── evaluate.py                    # Test set evaluation, confusion matrix, and classification report
├── predict.py                     # Single-image command-line inference script
├── data/
│   ├── README.md                  # Dataset download links, license, and structure guide
│   ├── raw/                       # Place HAM10000 images and metadata here (git-ignored)
│   └── processed/                 # Generated patient-split CSV files (train/val/test)
├── models/
│   ├── README.md                  # Model weights documentation and generation instructions
│   ├── skin_cancer_cnn.keras      # Generated trained model file (git-ignored)
│   └── class_indices.json         # Generated label-to-index mapping
├── reports/
│   ├── README.md                  # Guidelines for evaluation outputs
│   ├── classification_report.txt  # Generated test precision, recall, and F1-score
│   ├── metrics.json               # Generated test loss and accuracy values
│   └── figures/
│       ├── accuracy_loss_curves.png # Generated training vs validation curves
│       └── confusion_matrix.png     # Generated normalized confusion matrix
└── src/
    ├── __init__.py                # Package initialization
    ├── dataset.py                 # Dataset loader, patient split, class weights & preprocessing
    ├── model.py                   # 4-block CNN architecture, augmentation, and compile logic
    └── utils.py                   # Plotting routines, metrics serialization, and helpers
```

---

## ⚙️ Installation & Windows Setup Guide

Follow these exact commands in **Windows PowerShell** (or the VS Code Integrated Terminal).

### 1. Check Python Version
Ensure you have Python 3.10 or 3.11 installed:
```powershell
python --version
```
*(If Python is not recognized, install Python 3.11 from python.org or run `winget install Python.Python.3.11`).*

### 2. Navigate to Project Directory
```powershell
cd "C:\Users\Akshat Pratap\.gemini\antigravity\scratch\skin-cancer-classification-cnn"
```

### 3. Create a Virtual Environment
A virtual environment keeps project dependencies isolated from the rest of your computer:
```powershell
python -m venv venv
```

### 4. Activate the Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```
> **Troubleshooting PowerShell Script Execution**:  
> If Windows displays an error about `Execution_Policies`, run this one-time command in PowerShell and re-activate:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> .\venv\Scripts\Activate.ps1
> ```
> When activated, your prompt will show `(venv)`.

### 5. Upgrade pip and Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Execution Guide

### Step 1: Download & Prepare the Dataset
1. Download HAM10000 from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).
2. Place `HAM10000_metadata.csv` into `data/raw/`.
3. Extract `HAM10000_images_part_1` and `HAM10000_images_part_2` into `data/raw/`.

### Step 2: Train the CNN Model
Run the training script:
```powershell
# Standard training (15 epochs)
python train.py --epochs 15 --batch-size 32

# Or a quick dry-run test (5 epochs on 200 samples)
python train.py --epochs 5 --batch-size 16 --sample-limit 200
```
This automatically:
- Splits data by `lesion_id` and saves `data/processed/train.csv`, `val.csv`, and `test.csv`.
- Trains the CNN with real-time augmentation and balanced class weights.
- Saves the best model to `models/skin_cancer_cnn.keras`.
- Saves loss and accuracy plots to `reports/figures/accuracy_loss_curves.png`.

### Step 3: Evaluate on Held-Out Test Set
Evaluate the trained model against never-before-seen test images:
```powershell
python evaluate.py
```
This produces:
- `reports/classification_report.txt` (per-class Precision, Recall, F1)
- `reports/figures/confusion_matrix.png` (Confusion Matrix Heatmap)
- `reports/metrics.json` (Overall test accuracy and loss)

### Step 4: Predict on a Single Image (CLI)
Test inference on any skin image from your command line:
```powershell
python predict.py --image data/raw/HAM10000_images_part_1/ISIC_0024306.jpg
```

### Step 5: Launch the Streamlit Web Application
Run the browser dashboard:
```powershell
streamlit run app.py
```
Streamlit will launch automatically in your default web browser at `http://localhost:8501`.

---

## 📈 Evaluation Metrics & Integrity Policy

> **Academic Integrity Statement**:  
> In accordance with ethical engineering practices, this repository **does not include fabricated accuracy numbers** or claim unrealistic performance (e.g. "99% accuracy").  
> Real-world dermatoscopic datasets present severe visual overlap and extreme class imbalance. Genuine test metrics are calculated dynamically when you execute `evaluate.py`.

Once you run `evaluate.py`, report your actual recorded numbers below:
```text
Test Loss                : [Insert measured loss from reports/metrics.json]
Overall Test Accuracy    : [Insert measured accuracy from reports/metrics.json]
Macro-Averaged F1-Score  : [Insert macro avg F1 from reports/classification_report.txt]
```

---

## 📸 Adding Genuine Screenshots

To make your GitHub repository stand out to recruiters and professors, capture and embed real screenshots:

1. Launch your Streamlit application: `streamlit run app.py`.
2. Upload a test image and run classification.
3. Take a screenshot using the Windows Snipping Tool (`Windows Key + Shift + S`).
4. Save the screenshot to a new folder: `docs/screenshots/app_prediction.png`.
5. Link it in your README:
   ```markdown
   ![Streamlit Web Application Preview](docs/screenshots/app_prediction.png)
   ```

---

## ⚠️ Project Limitations

1. **Dataset Bias**: HAM10000 contains images predominantly from Caucasian populations with fair skin types (Fitzpatrick skin types I–III). Performance on darker skin tones (Fitzpatrick types IV–VI) may be degraded.
2. **Artifact Sensitivity**: Dermatoscopic images often contain surgical markings, hair, dark borders (vignetting), and air bubbles, which can influence CNN convolutional feature maps.
3. **Class Imbalance**: Despite class weighting, rare classes (`df`, `vasc`) have fewer than 150 training samples, making recall on those specific classes lower than on common moles (`nv`).
4. **Resolution Constraints**: Images are downscaled to 128x128 pixels to allow smooth CPU training on student laptops, discarding fine micro-structures present in original high-resolution (600x450) images.

---

## ⚖️ Medical & Ethical Disclaimer

```text
========================================================================================
                                MEDICAL DISCLAIMER
========================================================================================
This software is developed solely as an educational academic project by undergraduate
engineering students. It is NOT certified as a medical diagnostic tool or clinical decision
support software by any regulatory body (such as the US FDA, CE, or CDSCO).

Under no circumstances should this software be utilized for self-diagnosis, clinical
triage, or real-world medical decision-making. Deep neural networks evaluate mathematical
similarity to training pixels and can produce false positives and false negatives.

Any changing, irregular, bleeding, or suspicious skin lesion requires prompt physical
examination and histopathological biopsy by a licensed dermatologist or medical professional.
========================================================================================
```

---

## 🔮 Future Improvements

- [ ] **Transfer Learning**: Benchmark advanced pre-trained architectures (e.g., EfficientNetV2, MobileNetV3, ResNet50).
- [ ] **Explainable AI (XAI)**: Integrate Grad-CAM (Gradient-weighted Class Activation Mapping) to highlight which visual regions influenced the CNN's decision.
- [ ] **Diverse Datasets**: Evaluate against the ISIC 2024 / BDD100k multi-ethnic datasets to improve fairness across diverse skin tones.
- [ ] **Hair Removal Preprocessing**: Implement DullRazor morphological filtering to clean skin hair artifacts prior to CNN ingestion.

---

## 🌐 GitHub & LinkedIn Sharing Guide

### Step 1: Uploading to GitHub

#### Option A: Using the GitHub Web Interface (Simplest for beginners)
1. Go to [GitHub.com](https://github.com) and log in.
2. Click **New Repository** (green button).
3. Name it: `skin-cancer-classification-cnn`.
4. Leave "Initialize with README" **unchecked** (we already have a complete README).
5. Click **Create repository**.
6. Use Git commands in your PowerShell terminal:
   ```powershell
   git init
   git add .
   git commit -m "Initial commit: complete skin cancer CNN classification project"
   git branch -M main
   git remote add origin https://github.com/<YOUR-USERNAME>/skin-cancer-classification-cnn.git
   git push -u origin main
   ```

#### Option B: Using VS Code
1. Open the project folder in VS Code (`File > Open Folder`).
2. Click the **Source Control** icon on the left sidebar (or press `Ctrl + Shift + G`).
3. Click **Publish to GitHub**.
4. Select "Publish to GitHub Public Repository".

*(Note: Thanks to our `.gitignore`, heavy datasets and model binaries will not be uploaded, keeping your repository within GitHub size limits).*

---

### Step 2: Sharing on LinkedIn

When sharing your project on LinkedIn, highlight your engineering process, data leakage prevention, and academic integrity:

> **Sample LinkedIn Post:**
> 
> 🚀 Excited to share my first deep learning project: **Skin Cancer Classification Using Convolutional Neural Networks (CNN) & Python**!
> 
> As a first-year Computer Science Engineering student, I wanted to build an end-to-end computer vision pipeline from scratch rather than just running a tutorial.
> 
> 🔍 **Key Highlights:**
> • Built a 4-block custom CNN using TensorFlow and Keras to classify dermatoscopic images into 7 classes (HAM10000 dataset).  
> • **Data Integrity**: Addressed a critical data leakage trap by grouping on patient/lesion identifiers (`lesion_id`) rather than naive random image splitting.  
> • **Class Imbalance**: Handled extreme class skew (67% common moles vs 1% rare lesions) using balanced class weights.  
> • **Deployment**: Deployed an interactive Streamlit web dashboard displaying predicted classes, full confidence breakdowns, and clinical disclaimers.  
> 
> 🔗 GitHub Repository: [Link to your GitHub repo]  
> 
> *Note: This project is strictly educational and not a medical diagnostic device.*  
> 
> #DeepLearning #TensorFlow #ComputerVision #Python #Streamlit #MachineLearning #BTech #Engineering #AcademicProject

---

## 👤 Author & Attribution

- **Developer**: Akshat Pratap
- **Degree**: BTech Computer Science & Engineering (1st Year)
- **Dataset Attribution**: Tschandl P., Rosendahl C., Kittler H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Sci Data 5, 180161 (2018).

---

## 📄 License

- **Source Code**: Released under the [MIT License](LICENSE).
- **Dataset**: The HAM10000 dataset is governed by its original [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license.
