# 🏙️ Area-Specific Property Price Prediction in Dubai

This repository contains the complete codebase, notebooks, and documentation for the master's project: **"Area-Specific Property Price Prediction in Dubai Using Property Attributes and Location Characteristics"**, developed as part of the MSc in Data Science at Cardiff Metropolitan University.

## 📌 Project Overview

The goal of this project is to build an accurate and interpretable machine learning model to predict property prices in Dubai, leveraging features such as area, property size, proximity to landmarks, and property usage. The final model is deployed via an interactive Streamlit dashboard for real-time prediction and explanation.

---

## 🔍 Key Features

- **Machine Learning Models**: Random Forest, XGBoost, Linear Regression
- **Localized Modeling**: Separate models per neighborhood (`area_name_en`)
- **Feature Engineering**: Handling missing values, outliers, encoding, and interaction features
- **EDA**: Correlation analysis, ANOVA, and Cramér’s V for deep feature insights
- **Model Interpretability**: Feature importance plots
- **Deployment**: Streamlit dashboard with dynamic user input and prediction display

---

## 📁 Repository Structure

```plaintext
📦 Project Root
├── data/
│   └── [Note: Raw dataset not included due to file size]
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_area_specific_models.ipynb
│   └── 04_streamlit_dashboard_preview.ipynb
├── models_by_area/
│   └── *.pkl (trained models per area)
├── streamlit_app/
│   └── app.py (Streamlit dashboard)
├── visuals/
│   └── *.png (EDA and model evaluation graphs)
├── README.md
└── requirements.txt
