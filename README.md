# 🎓 Student Performance Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/yourusername/student-performance-predictor/graphs/commit-activity)

> A machine learning-powered system for predicting student performance bands based on test-taking patterns. Built with Python, scikit-learn, and Streamlit.

[Live Demo](#) | [Report Bug](https://github.com/yourusername/student-performance-predictor/issues) | [Request Feature](https://github.com/yourusername/student-performance-predictor/issues)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Dataset Analysis](#-dataset-analysis)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Results & Insights](#-results--insights)
- [Screenshots](#-screenshots)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

This project analyzes GLT 302 test results from 199 Microbiology students and builds a predictive model to classify student performance into seven bands. The system helps educators identify at-risk students early and provide targeted interventions.

### 🚨 Problem Statement

The original dataset revealed:
- **82.41% failure rate** - Only 35 out of 199 students passed
- **Average score of 34.07%** - Significantly below passing threshold
- **40% scored below 20%** - Critical performance requiring immediate intervention
- **Wide performance variance** - Scores ranging from 0% to 90%

### 💡 Solution

An end-to-end machine learning pipeline that:
1. Analyzes test-taking patterns and performance correlations
2. Predicts student performance bands with 70% accuracy
3. Provides real-time risk assessment through an interactive web dashboard
4. Generates personalized intervention recommendations

---

## ✨ Key Features

### 📊 Comprehensive Data Analysis
- Descriptive statistics and distribution analysis
- Department-wise performance comparison (Morning vs Evening classes)
- Temporal pattern analysis (submission time vs. performance)
- Score distribution across 7 performance bands
- Multiple attempt detection and analysis

### 🤖 Machine Learning Model
- **Multi-class classification** into 7 performance bands
- **Random Forest & Gradient Boosting** algorithms
- **Cross-validation** for robust evaluation
- **Feature importance analysis** to identify key predictors
- **Class imbalance handling** with balanced weights

### 🌐 Interactive Web Application
- User-friendly **Streamlit interface**
- **Real-time predictions** with confidence scores
- **Interactive visualizations** using Plotly
- **Personalized recommendations** based on risk levels
- **Model insights dashboard** with feature importance

---

## 📊 Dataset Analysis

### Dataset Overview

| Metric | Value |
|--------|-------|
| **Total Records** | 199 test submissions |
| **Unique Students** | 188 |
| **Departments** | 2 (Morning & Evening Classes) |
| **Test Date** | March 17, 2026 |
| **Time Range** | 6:31 AM - 9:18 PM |
| **Score Range** | 0% - 90% |
| **Average Score** | 34.07% |
| **Median Score** | 30.0% |
| **Pass Rate** | 17.59% |


### Key Insights

#### 🏫 Department Comparison

| Department | Students | Pass Rate | Avg Score | Median | Max Score |
|------------|----------|-----------|-----------|--------|-----------|
| **Morning Class** | 132 (66.33%) | 19.70% | 34.85% | 30.0% | 90% |
| **Evening Class** | 67 (33.67%) | 13.43% | 32.54% | 32.5% | 85% |

**Finding:** Morning class outperforms evening class by 6.27 percentage points in pass rate.

#### ⏰ Temporal Performance Patterns

| Time Period | Submissions | Average Score | Best Time |
|-------------|-------------|---------------|-----------|
| **6 AM** | 12 | 5.42% | ❌ Worst |
| **7 AM** | 54 | 23.84% | ⚠️ Poor |
| **8-9 AM** | 46 | 40.22% | ⚡ Moderate |
| **10-11 AM** | 13 | 56.88% | ✅ Best |
| **12-5 PM** | 52 | 40.15% | ⚡ Moderate |
| **6-9 PM** | 22 | 33.75% | ⚠️ Below Average |

**Finding:** Students submitting at 10-11 AM scored **10x higher** than those at 6 AM.

#### 🔄 Multiple Attempts

4 students took multiple attempts:
- **Ajiboyede Roseline oluwayemisi:** 9 attempts (all scored 27.5%)
- **Ojo Precious Olayinka:** 2 attempts
- **Oladimeji iyanuoluwa bolanle:** 2 attempts
- **Salami Halimat Damilola:** 2 attempts (improved from 30% to 52.5%)

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/student-performance-predictor.git
cd student-performance-predictor

# 2. Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python --version
streamlit --version

============================================================
SCORE RANGE PREDICTION MODEL TRAINING
============================================================

Dataset Size: 199 records
Training Set: 159 | Test Set: 40

Performance Band Distribution:
Poor                45
Critical            43
Below Average       35
Very Poor           37
Average             16
Good                12
Excellent           11

============================================================
Training Random Forest Classifier...
============================================================

Random Forest Accuracy: 0.6750

Classification Report:
              precision    recall  f1-score   support

     Average       0.67      0.50      0.57         4
Below Average       0.71      0.71      0.71         7
    Critical       0.75      0.86      0.80         7
   Excellent       1.00      0.67      0.80         3
        Good       0.50      0.67      0.57         3
        Poor       0.60      0.60      0.60         10
   Very Poor       0.60      0.50      0.55         6

    accuracy                           0.68        40
   macro avg       0.69      0.64      0.66        40
weighted avg       0.68      0.68      0.67        40

Cross-Validation Scores: [0.59 0.66 0.63 0.59 0.66]
Mean CV Score: 0.6260 (+/- 0.0612)

Feature Importance:
           feature  importance
  submission_hour    0.452341
       Department    0.298765
     time_of_day     0.156234
     num_attempts    0.092660

============================================================
BEST MODEL: Random Forest (Accuracy: 0.6750)
============================================================

✓ Model saved as 'score_predictor_model.pkl'
✓ Encoders saved successfully

============================================================
MODEL TRAINING COMPLETE!
============================================================

╔════════════════════════════════════════════════════════╗
║              MODEL PERFORMANCE METRICS                 ║
╠════════════════════════════════════════════════════════╣
║  Overall Accuracy:           67.50%                    ║
║  Cross-Validation Score:     62.60% (±6.12%)          ║
║  Training Samples:           159                       ║
║  Test Samples:               40                        ║
║  Number of Features:         4                         ║
║  Number of Classes:          7                         ║
╚════════════════════════════════════════════════════════╝

📊 Feature Contribution to Predictions:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Submission Hour    ████████████████████████████  45.23%
Department         ████████████████████          29.88%
Time of Day        ████████                      15.62%
Num Attempts       █████                          9.27%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📂Project Structure:

student-performance-predictor/
│
├── 📂 data/
│   ├── GLT_302_TEST_ADENIYI_BATCH_1.csv    # Original dataset
│   └── README.md                            # Data documentation
│
├── 📂 models/
│   ├── score_predictor_model.pkl           # Trained Random Forest model
│   ├── label_encoder_dept.pkl              # Department encoder
│   ├── label_encoder_time.pkl              # Time period encoder
│   ├── label_encoder_target.pkl            # Target variable encoder
│   └── feature_names.txt                   # Feature reference
│
├── 📂 notebooks/
│   ├── 01_exploratory_analysis.ipynb       # EDA and visualizations
│   ├── 02_feature_engineering.ipynb        # Feature creation
│   └── 03_model_experiments.ipynb          # Model comparison
│
├── 📂 src/
│   ├── __init__.py
│   ├── train_model.py                      # Model training script
│   
