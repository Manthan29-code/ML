import joblib
import numpy as np
import pandas as pd
import streamlit as st

FEATURE_ORDER = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

CATEGORICAL_LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {
        0: "Typical angina",
        1: "Atypical angina",
        2: "Non-anginal pain",
        3: "Asymptomatic",
    },
    "fbs": {0: "<= 120 mg/dl", 1: "> 120 mg/dl"},
    "restecg": {
        0: "Normal",
        1: "ST-T wave abnormality",
        2: "Left ventricular hypertrophy",
    },
    "exang": {0: "No", 1: "Yes"},
    "slope": {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
    "ca": {0: "0", 1: "1", 2: "2", 3: "3"},
    "thal": {1: "Normal", 2: "Fixed defect", 3: "Reversible defect"},
}

st.set_page_config(
    page_title="Heart Health Predictor",
    page_icon="\u2764",
    layout="wide",
)

CSS = """
<style>
:root {
    --bg: #fff7ed;
    --surface: #ffffff;
    --primary: #0d9488;
    --accent: #f97316;
    --text: #1f2937;
}

.stApp {
    background: radial-gradient(circle at 10% 10%, #fff1e6 0%, #fff7ed 45%, #fef3c7 100%);
    color: var(--text);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2.5rem;
}

.header-card {
    background: linear-gradient(135deg, #0d9488 0%, #14b8a6 55%, #f97316 130%);
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 18px;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.15);
    margin-bottom: 1.5rem;
}

.panel {
    background: var(--surface);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
    border: 1px solid rgba(148, 163, 184, 0.2);
}

div[data-testid="stVerticalBlock"]:has(.panel-marker) {
    background: var(--surface);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
    border: 1px solid rgba(148, 163, 184, 0.2);
}

.panel-marker {
    display: none;
}

.result-card {
    background: linear-gradient(145deg, #ffffff 0%, #fff7ed 100%);
    border-left: 6px solid var(--accent);
    padding: 1.2rem 1.4rem;
    border-radius: 14px;
    box-shadow: 0 10px 22px rgba(15, 23, 42, 0.12);
}

.badge {
    display: inline-block;
    background: var(--primary);
    color: white;
    padding: 0.35rem 0.65rem;
    border-radius: 999px;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
}

div.stButton > button {
    background: var(--primary);
    color: white;
    border-radius: 999px;
    padding: 0.6rem 1.4rem;
    border: none;
    font-weight: 600;
    box-shadow: 0 10px 20px rgba(13, 148, 136, 0.25);
}

div.stButton > button:hover {
    background: #0f766e;
    color: white;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_artifacts(model_path: str, scaler_path: str):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def get_numeric_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
        stats[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
        }
    return stats


def build_feature_row(values: dict) -> pd.DataFrame:
    ordered = [values[name] for name in FEATURE_ORDER]
    return pd.DataFrame([ordered], columns=FEATURE_ORDER)


st.markdown(
    """
    <div class="header-card">
        <h1>Heart Health Predictor</h1>
        <p>Enter patient details to estimate heart disease risk using the trained model.</p>
        <span class="badge">Decision Tree + StandardScaler</span>
    </div>
    """,
    unsafe_allow_html=True,
)

data_path = "heart.csv"
model_path = "Decision_Tree.pkl"
scaler_path = "scaler.pkl"

try:
    data = load_data(data_path)
    numeric_stats = get_numeric_stats(data)
except FileNotFoundError:
    st.error("heart.csv not found in the app directory.")
    st.stop()

try:
    model, scaler = load_artifacts(model_path, scaler_path)
except FileNotFoundError:
    st.error("Model files not found. Ensure Decision_Tree.pkl and scaler.pkl are present.")
    st.stop()

left, right = st.columns([1.05, 1])

with left:
    st.markdown("<div class='panel-marker'></div>", unsafe_allow_html=True)
    st.subheader("Patient Inputs")

    age = st.number_input(
        "Age",
        min_value=int(numeric_stats["age"]["min"]),
        max_value=int(numeric_stats["age"]["max"]),
        value=int(numeric_stats["age"]["median"]),
        step=1,
    )

    trestbps = st.number_input(
        "Resting blood pressure (trestbps)",
        min_value=int(numeric_stats["trestbps"]["min"]),
        max_value=int(numeric_stats["trestbps"]["max"]),
        value=int(numeric_stats["trestbps"]["median"]),
        step=1,
    )

    chol = st.number_input(
        "Serum cholesterol (chol)",
        min_value=int(numeric_stats["chol"]["min"]),
        max_value=int(numeric_stats["chol"]["max"]),
        value=int(numeric_stats["chol"]["median"]),
        step=1,
    )

    thalach = st.number_input(
        "Maximum heart rate achieved (thalach)",
        min_value=int(numeric_stats["thalach"]["min"]),
        max_value=int(numeric_stats["thalach"]["max"]),
        value=int(numeric_stats["thalach"]["median"]),
        step=1,
    )

    oldpeak = st.slider(
        "ST depression induced by exercise (oldpeak)",
        min_value=float(numeric_stats["oldpeak"]["min"]),
        max_value=float(numeric_stats["oldpeak"]["max"]),
        value=float(numeric_stats["oldpeak"]["median"]),
        step=0.1,
    )

    st.markdown("---")

    sex_label = st.selectbox("Sex", list(CATEGORICAL_LABELS["sex"].values()))
    cp_label = st.selectbox("Chest pain type (cp)", list(CATEGORICAL_LABELS["cp"].values()))
    fbs_label = st.selectbox("Fasting blood sugar (fbs)", list(CATEGORICAL_LABELS["fbs"].values()))
    restecg_label = st.selectbox(
        "Resting electrocardiographic results (restecg)",
        list(CATEGORICAL_LABELS["restecg"].values()),
    )
    exang_label = st.selectbox("Exercise induced angina (exang)", list(CATEGORICAL_LABELS["exang"].values()))
    slope_label = st.selectbox("Slope of peak exercise ST segment (slope)", list(CATEGORICAL_LABELS["slope"].values()))
    ca_label = st.selectbox("Number of major vessels (ca)", list(CATEGORICAL_LABELS["ca"].values()))
    thal_label = st.selectbox("Thalassemia (thal)", list(CATEGORICAL_LABELS["thal"].values()))

with right:
    st.markdown("<div class='panel-marker'></div>", unsafe_allow_html=True)
    st.subheader("Prediction")

    predict = st.button("Predict Heart Disease")

    if predict:
        label_to_value = {
            key: {label: value for value, label in mapping.items()}
            for key, mapping in CATEGORICAL_LABELS.items()
        }

        values = {
            "age": int(age),
            "sex": label_to_value["sex"][sex_label],
            "cp": label_to_value["cp"][cp_label],
            "trestbps": int(trestbps),
            "chol": int(chol),
            "fbs": label_to_value["fbs"][fbs_label],
            "restecg": label_to_value["restecg"][restecg_label],
            "thalach": int(thalach),
            "exang": label_to_value["exang"][exang_label],
            "oldpeak": float(oldpeak),
            "slope": label_to_value["slope"][slope_label],
            "ca": label_to_value["ca"][ca_label],
            "thal": label_to_value["thal"][thal_label],
        }

        feature_row = build_feature_row(values)
        scaled = scaler.transform(feature_row)
        prediction = model.predict(scaled)[0]

        probability = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(scaled)[0][1]

        label = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown(f"### {label}")
        if probability is not None:
            st.write(f"Probability: {probability:.2%}")
        else:
            st.write("Probability not available for this model.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Adjust inputs and run a prediction.")

st.caption("Model outputs are for educational use and should not replace clinical advice.")
