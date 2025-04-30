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
├── Required_Files/
│   └── [Note: Raw dataset not included due to file size]
├── Code/
│   ├── Data_Cleaning_01.ipynb
│   ├── Handling_Outliers_02.ipynb
│   ├── Modal_Training_03.ipynb
│   └── Modal_Training_Group_by_Area.ipynb
├── models_by_area/
│   └── *.pkl (trained models per area)
├── streamlit_app/
│   └── app.py (Streamlit dashboard)
├── visuals/
│   └── *.png (EDA and model evaluation graphs)
├── README.md
```

---

## ⚠️ Data Availability

> **Note**: Due to GitHub's file size limitations, the raw dataset (`~1M+ records`) from Dubai Pulse is **not included** in this repository.

To replicate the results:

1. Visit: [https://www.dubaipulse.gov.ae/](https://www.dubaipulse.gov.ae/)
2. Search for **Dubai Land Department Property Transactions** dataset.
3. Download and save the data in the `/data/` directory.
4. Follow the preprocessing steps in the notebooks to clean and transform the data.

---

## 🚀 Running the Streamlit Dashboard

```bash
cd streamlit_app
streamlit run app.py
```

The dashboard will allow you to:
- Select an area
- Input property details
- View predicted price and contributing features

---

## 📊 OverModel Performance Summary

| Metric        | Value    |
|---------------|----------|
| R² (Overall)  | 0.82     |
| RMSE          | 4,049.93 |
| MAE           | 3,496.35 |

### 🔹 Area-Specific Performance (Sample)

| Area                  | R²   | RMSE        | MAE         |
|-----------------------|------|--------------|------------|
| Business Bay          | 0.98 | 851,844.26   | 339,888.01 |
| Zaabeel First         | 0.96 | 581,221.96   | 344,742.81 |
| Al Barsha South Fifth | 0.78 | 185,739.39   | 109,956.81 |
| Palm Jumeirah         | 0.75 | 2,569,151.96 | 886,473.72 |
| Marsa Dubai           | 0.36 | 2,110,218.34 | 500,908.40 |

> Some areas show lower R² due to complex price variance or sparse data. Area-wise modeling significantly improved overall accuracy and interpretability.


---

## 🔮 Future Improvements

- Integrate SHAP for deeper explainability
- Add time-based analysis (seasonality, holidays)
- Extend dashboard with historical trends

---

## 🧑‍💻 Author

**Chamodi Rasandika Dilrukshi Rajapakshe**  
MSc Data Science – Cardiff Metropolitan University

---

## 📃 License

This project is licensed under the [MIT License](LICENSE). Data is sourced from the Dubai Pulse Open Data initiative.

