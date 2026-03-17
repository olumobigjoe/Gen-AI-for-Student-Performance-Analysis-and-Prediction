# 🎓 Student Performance Prediction System

A machine learning-powered system for predicting student performance bands based on test-taking patterns. Built with Python, scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Results & Insights](#results--insights)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

This project analyzes GLT 302 test results from Microbiology students and builds a predictive model to classify student performance into seven bands (Excellent, Good, Average, Below Average, Poor, Very Poor, Critical). The system helps educators identify at-risk students early and provide targeted interventions.

### Key Highlights

- **82.41% failure rate** in the original dataset
- **Random Forest Classifier** for multi-class prediction
- **Interactive Streamlit dashboard** for real-time predictions
- **Comprehensive data analysis** with actionable insights

## ✨ Features

### Data Analysis
- ✅ Comprehensive descriptive statistics
- ✅ Department-wise performance comparison
- ✅ Temporal pattern analysis (submission time vs. performance)
- ✅ Score distribution visualization
- ✅ Multiple attempt detection

### Machine Learning Model
- ✅ Multi-class classification (7 performance bands)
- ✅ Random Forest & Gradient Boosting algorithms
- ✅ Cross-validation for robust evaluation
- ✅ Feature importance analysis
- ✅ Class imbalance handling

### Web Application
- ✅ User-friendly Streamlit interface
- ✅ Real-time predictions with confidence scores
- ✅ Interactive visualizations
- ✅ Personalized recommendations
- ✅ Model insights dashboard

## 📊 Dataset

**Source:** GLT_302_TEST_ADENIYI_BATCH_1.csv

### Dataset Statistics
- **Total Records:** 199 test submissions
- **Unique Students:** 188
- **Departments:** 2 (Morning & Evening Classes)
- **Test Date:** March 17, 2026
- **Score Range:** 0% - 90%
- **Average Score:** 34.07%

### Features
| Feature | Type | Description |
|---------|------|-------------|
| Timestamp | DateTime | Test submission time |
| Name | String | Student name |
| App_No | Integer | Application number |
| Department | Categorical | Morning/Evening class |
| Score (%) | Float | Test score percentage |
| Result | Categorical | PASS/FAIL |

### Performance Band Distribution
