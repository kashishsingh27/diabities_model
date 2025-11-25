# ==============================
# AI Diabetes Predictor Dashboard
# ==============================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(page_title="AI Diabetes Predictor Dashboard", layout="wide")

# ------------------------------
# Custom Styling
# ------------------------------
st.markdown("""
<style>
    .stTabs [data-baseweb="tab"] {
        color: #1E90FF;
        background-color: #F0F8FF;
        border-radius: 10px;
        padding: 10px 20px;
        margin-right: 8px;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        color: white !important;
        background-color: #0A3D62 !important;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Load Data and Model
# ------------------------------
@st.cache_data
def load_evaluation():
    return pd.read_csv("models/model_evaluation.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/best_model.joblib")  # contains {"model": ..., "scaler": ...}

try:
    results = load_evaluation()
    loaded = load_model()
    model = loaded["model"]
    scaler = loaded["scaler"]

except Exception as e:
    st.error(f"Error loading model or files: {e}")
    st.stop()

# ------------------------------
# Tabs
# ------------------------------
tabs = st.tabs(["🔮 Live Prediction", "📊 Model Evaluation", "📈 Data Insights"])

# ==============================
# TAB 1 — LIVE PREDICTION
# ==============================
with tabs[0]:
    st.markdown("## 🔮 Live Diabetes Prediction")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 👤 Patient Demographics")
        Pregnancies = st.number_input("Pregnancies", 0, 20, 2)
        Age = st.number_input("Age", 1, 120, 30)

        st.markdown("### 🩺 Vital Signs")
        Glucose = st.number_input("Glucose Level", 0, 300, 120)
        BloodPressure = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70)

    with col2:
        st.markdown("### 📏 Body Measurements")
        BMI = st.number_input("BMI (kg/m²)", 0.0, 70.0, 25.0)
        SkinThickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)

        st.markdown("### 🧬 Clinical Data")
        Insulin = st.number_input("Insulin Level (μU/mL)", 0, 900, 80)
        DPF = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)

    st.markdown("---")

    # =====================
    # 🔍 PREDICTION LOGIC
    # =====================
    if st.button("🔍 Predict Diabetes Risk", use_container_width=True):
        try:
            user_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness,
                          Insulin, BMI, DPF, Age]]

            # Scale input
            scaled_data = scaler.transform(user_data)

            # Get probability (risk %)
            proba = model.predict_proba(scaled_data)[0][1]
            risk_percent = round(proba * 100, 2)

            # Predict class (0 or 1)
            prediction = model.predict(scaled_data)[0]

            # Show Risk %
            st.metric("Diabetes Risk (%)", f"{risk_percent}%")

            # Risk progress bar
            st.progress(proba)

            # Final prediction message
            if int(prediction) == 1:
                st.error(f"### 🔴 High Risk: {risk_percent}% chance of Diabetes")
            else:
                st.success(f"### 🟢 Low Risk: {risk_percent}% chance of Diabetes")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ==============================
# TAB 2 — MODEL EVALUATION
# ==============================
with tabs[1]:
    st.markdown("## 📊 Model Evaluation Metrics")
    st.markdown("---")
    
    st.dataframe(results, use_container_width=True)

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    for i in range(0, len(metrics), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(metrics):
                metric = metrics[i + j]
                if metric in results.columns:
                    with cols[j]:
                        fig, ax = plt.subplots(figsize=(7,6))
                        bars = sns.barplot(
                            x="Model", y=metric, data=results, palette="viridis", ax=ax
                        )
                        ax.set_ylim(0, 1)
                        ax.set_title(f"{metric} Comparison", fontsize=16, fontweight='bold')
                        ax.tick_params(axis='x', rotation=45)
                        ax.grid(axis='y', alpha=0.3)

                        for bar in bars.patches:
                            height = bar.get_height()
                            ax.text(
                                bar.get_x() + bar.get_width()/2., height,
                                f"{height:.3f}", ha='center', va='bottom'
                            )

                        st.pyplot(fig)

    st.markdown("---")
    if "Accuracy" in results.columns:
        best_model_name = results.loc[results["Accuracy"].idxmax(), "Model"]
        best_accuracy = results["Accuracy"].max()
        st.success(f"### 🏆 Best Model: **{best_model_name}** (Accuracy: {best_accuracy:.3f})")

# ==============================
# TAB 3 — DATA INSIGHTS
# ==============================
with tabs[2]:
    st.markdown("## 📈 Data Insights & Comparisons")
    st.markdown("---")

    try:
        df = pd.read_csv("diabetes.csv")
    except:
        st.error("Could not find diabetes.csv.")
        st.stop()

    # Correlation Heatmap
    st.subheader("🔥 Feature Correlation Heatmap")
    corr = df.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        labels=dict(color="Correlation")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Feature Distributions
    st.subheader("📊 Key Feature Distributions by Outcome")
    key_features = ["Glucose", "BMI", "Age", "Insulin"]

    for i in range(0, len(key_features), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(key_features):
                col_name = key_features[i + j]
                with cols[j]:
                    fig, ax = plt.subplots(figsize=(6,4))

                    sns.histplot(df[df['Outcome'] == 0][col_name],
                                 label="Non-Diabetic", alpha=0.6, kde=True, ax=ax)
                    sns.histplot(df[df['Outcome'] == 1][col_name],
                                 label="Diabetic", alpha=0.6, kde=True, ax=ax)

                    ax.set_title(f"{col_name} Distribution")
                    ax.legend()
                    st.pyplot(fig)
