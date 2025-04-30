import streamlit as st
import pandas as pd
import joblib
import os

# Load processed dataset for dropdown values
data = pd.read_csv("Outlier_Filtered_Dataset.csv")

# Title
st.title("Real Estate Price Estimator")

# Sidebar header
st.sidebar.header("User Input Features")

# Helper: Dropdown options from dataset
def get_unique_values(column):
    return sorted(data[column].dropna().unique().tolist())

# Grouped data for easier lookups
area_groups = data.groupby("area_name_en")

# User selects area name first
area_name = st.sidebar.selectbox("Area Name", get_unique_values("area_name_en"))
area_data = area_groups.get_group(area_name)

# Load model based on selected area
model_path = f"models_by_area/{area_name}.pkl"
if os.path.exists(model_path):
    model, model_columns = joblib.load(model_path)

    # Feature importance
    importances = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": model_columns, "Importance": importances})

    def is_relevant(feature_prefix):
        return any(importance_df["Feature"].str.startswith(feature_prefix) & (importance_df["Importance"] > 0.01))

    # Get default mode values
    def get_mode_value(col):
        return area_data[col].mode().iloc[0] if not area_data[col].dropna().empty else ""

    # Property Usage
    property_usage = st.sidebar.selectbox("Property Usage", get_unique_values("property_usage_en"), index=get_unique_values("property_usage_en").index(get_mode_value("property_usage_en")))

    # Procedure Area - Min procedure area not zero
    min_procedure_area = int(area_data[area_data['procedure_area'] > 9]['procedure_area'].min())
    max_procedure_area = int(area_data['procedure_area'].max())
    procedure_area = st.sidebar.slider("Procedure Area (Square Meters)", min_procedure_area, max_procedure_area, min_procedure_area, step=50)

    # Parking
    has_parking = st.sidebar.selectbox("Parking Availability", ["Yes", "No"], index=0 if get_mode_value("has_parking") == "Yes" else 1)

    # Smart dropdowns with importance awareness
    def smart_selectbox(label, column, feature_prefix):
        options = get_unique_values(column)
        default_value = get_mode_value(column)
        if not is_relevant(feature_prefix):
            return st.sidebar.selectbox(f"{label} (Not relevant for prediction)", ["Not relevant for prediction"], disabled=True)
        else:
            return st.sidebar.selectbox(label, options, index=options.index(default_value))

    nearest_landmark = smart_selectbox("Nearest Landmark", "nearest_landmark_en", "nearest_landmark_en")
    nearest_metro = smart_selectbox("Nearest Metro", "nearest_metro_en", "nearest_metro_en")
    nearest_mall = smart_selectbox("Nearest Mall", "nearest_mall_en", "nearest_mall_en")

    # Format input as DataFrame
    input_df = pd.DataFrame([{
        'has_parking': has_parking,
        'procedure_area': procedure_area,
        'nearest_landmark_en': nearest_landmark,
        'nearest_metro_en': nearest_metro,
        'nearest_mall_en': nearest_mall,
        'property_usage_en': property_usage
    }])

    # Encode input
    input_encoded = pd.get_dummies(input_df)

    # Align input with model
    for col in model_columns:
        if col not in input_encoded:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_columns]

    # Conditional prediction
    if procedure_area == 0:
        st.warning("Please increase the procedure area to see estimated property price.")
    else:
        prediction = model.predict(input_encoded)[0]
        st.subheader("Estimated Property Price:")
        st.success(f"{prediction:,.2f}")

        # Top contributing features
        st.markdown("### These are the features contribute for the Actual Worth in Selected Area:")
        important_feats = importance_df[importance_df["Importance"] > 0.01].sort_values(by="Importance", ascending=False)
        important_feats.reset_index(drop=True, inplace=True)
        important_feats.index += 1
        st.dataframe(important_feats)
else:
    st.error(f"Model for area '{area_name}' not found.")
