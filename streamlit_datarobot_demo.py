"""
Streamlit + DataRobot Interactive Prediction App

This app demonstrates how to embed a deployed DataRobot AutoML model into a lightweight Streamlit interface. Users can either upload a CSV of feature data or enter feature values manually, then see model predictions and feature impacts in real time.

---

## Prerequisites

1. **Python 3.8+** installed on your machine.  Verify with:
   ```bash
   python --version
   ```

2. **Install required packages**:
   ```bash
   pip install streamlit datarobot pandas
   ```

3. **DataRobot account & deployment**:
   - Sign up or log into your DataRobot workspace (https://app.datarobot.com).
   - Upload your dataset (e.g., your flights CSV) and create a project.
   - Run Autopilot to train models.
   - Select your best model and click **Deploy**.  Copy the **Deployment ID** from the Deployments page.
   - Go to **My Account → API Tokens**, generate or copy your **API token**.

4. **Configure Streamlit secrets**:
   In your project folder, create a folder named `.streamlit` and inside it a file `secrets.toml`:
   ```toml
   DR_API_TOKEN = "<YOUR_DATAROBOT_API_TOKEN>"
   DR_API_URL   = "https://app.datarobot.com/api/v2"
   DEPLOYMENT_ID = "<YOUR_DEPLOYMENT_ID>"
   ```

---

### Running the App

From your terminal in the directory containing this script, run:
```bash
streamlit run streamlit_datarobot_demo.py
```

A browser window will open with the app UI.

---

import streamlit as st
import pandas as pd
from datarobot import Client

# --- Initialize DataRobot client ---
API_TOKEN     = st.secrets["DR_API_TOKEN"]
API_URL       = st.secrets["DR_API_URL"]
DEPLOYMENT_ID = st.secrets["DEPLOYMENT_ID"]
client = Client(token=API_TOKEN, endpoint=API_URL)
deployment = client.get_deployment(DEPLOYMENT_ID)

# --- UI Layout ---
st.title("📊 DataRobot + Streamlit Interactive Demo")

st.markdown(
    "Upload a CSV of your features or enter values manually to see predictions from your DataRobot model."
)

mode = st.radio("Choose input mode:", ["Upload CSV", "Manual Entry"])

if mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Input Data Preview")
        st.dataframe(df.head())

        with st.spinner("Requesting predictions..."):
            predictions = deployment.predict(df).get_all_as_dataframe()

        st.subheader("Predictions")
        st.dataframe(predictions)

        if 'predictionFeatureImpacts' in predictions.columns:
            st.subheader("Feature Impacts")
            st.write(predictions[['prediction', 'predictionFeatureImpacts']])

else:
    st.subheader("Manual Entry of Feature Values")
    # TODO: Replace feature_names with your model's actual input column names
    feature_names = ["feature1", "feature2", "feature3"]
    input_dict = {}
    for feat in feature_names:
        input_dict[feat] = st.text_input(f"{feat}")

    if st.button("Predict"):
        df_manual = pd.DataFrame([input_dict])
        with st.spinner("Generating prediction..."):
            result_df = deployment.predict(df_manual).get_all_as_dataframe()
        st.subheader("Prediction Result")
        st.dataframe(result_df)

# --- Footer ---
st.markdown("---")
st.caption("App built by [Your Name]. Proof‑of‑work and detailed setup instructions posted in Brightspace.")
"""
