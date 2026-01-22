import os
import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Ice Cream Revenue — Dashboard", page_icon="🍦", layout="wide")

def header():
    st.markdown(
        """
        <div style="text-align: center; background-color: #ffe5cc; padding: 20px; border-radius: 10px;">
            <h1 style="color: #ff6600; font-size: 3em;">🍦 Ice Cream Revenue Dashboard 📈</h1>
            <p style="color: #4CAF50; font-size: 1.2em;">A simple, non-technical view of model results and live predictions</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

header()

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models') 

def models_exist():
    required = ['svm_model.joblib', 'scaler.joblib', 'results.joblib']
    return all(os.path.exists(os.path.join(MODEL_DIR, r)) for r in required)

@st.cache_data
def load_data(path="Ice_Cream.csv"):
    try:
        df = pd.read_csv(path)
        df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
        df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
        df = df.dropna()
        return df
    except FileNotFoundError:
        st.error(f"Data file not found: {path}. Please ensure Ice_Cream.csv is in the same directory as app.py")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

df = load_data()

st.markdown("---") 

st.markdown(
    """
    ## 💡 Understanding Our Predictor
    We trained two models to predict whether daily **revenue will be High or Low** based on the day's temperature. 
    
    **High** means revenue is at or above the historical median. The **SVM model** is recommended for live prediction here.
    """,
    unsafe_allow_html=True,
)
st.write("")

if not models_exist():
    st.error("Model artifacts not found. Run training_notebook.ipynb to create models/ before using this dashboard.")
    with st.expander("How to create models 🛠️"):
        st.markdown(
            "1. Open **training_notebook.ipynb** and run all cells.\n2. That creates `models/` with trained artifacts.\n3. Refresh this page."
        )
    st.stop()

# Import joblib only after checking if models exist
try:
    import joblib
except ImportError:
    st.error("Required package 'joblib' is not installed. Please install it using: pip install joblib")
    st.stop()

try:
    svm = joblib.load(os.path.join(MODEL_DIR, 'svm_model.joblib'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    results = joblib.load(os.path.join(MODEL_DIR, 'results.joblib'))
except Exception as e:
    st.error(f"Error loading models: {e}. Ensure training notebook was run successfully.")
    st.stop()

st.markdown("---")
st.subheader("📊 Key Performance Indicators")
col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 2])
with col1:
    st.metric(label="✅ SVM Accuracy (Test Set)", value=f"{results['svm']['accuracy']:.2%}")
with col2:
    st.metric(label="🌳 Decision Tree Accuracy (Test Set)", value=f"{results['decision_tree']['accuracy']:.2%}")
with col3:
    median_rev = df['Revenue'].median()
    st.metric(label="💰 Historical Median Revenue", value=f"NPR {median_rev:,.0f}") 
with col4:
    st.markdown(
        """
        **Live Prediction Goal:** Predict whether revenue will be **High** (>= median) or **Low** for a given temperature. Use the **interactive panel to the right** to try values.
        """, unsafe_allow_html=True)

st.markdown("---")

left, right = st.columns([2, 1])

with left:
    st.subheader("📈 Data Overview — Revenue vs. Temperature")
    
    df['TempRound'] = df['Temperature'].round(0).astype(int)
    agg = df.groupby('TempRound', as_index=False)['Revenue'].mean().rename(columns={'Revenue': 'AvgRevenue'})
    agg = agg.sort_values('TempRound')
    
    st.markdown("### Average Revenue by Temperature (°C)")
    st.bar_chart(data=agg.set_index('TempRound')['AvgRevenue'], color="#FF9900") 
    
    st.write("---")
    st.markdown("🔥 **How to read this**: Each bar shows the **average revenue** observed historically at that temperature (rounded). **Higher bars** mean days at that temperature tended to bring more revenue.")

    st.write("")
    st.subheader("🌡️ Scatter Plot: Temperature vs. Revenue (Sample)")
    sample = df.sample(min(300, len(df)), random_state=1)
    
    sample['Label'] = (sample['Revenue'] >= median_rev).astype(int)
    
    spec = {
        "mark": "point",
        "encoding": {
            "x": {"field": "Temperature", "type": "quantitative", "title": "Temperature (°C)"},
            "y": {"field": "Revenue", "type": "quantitative", "title": "Revenue"},
            "color": {"field": "Label", "type": "nominal", "title": "High Revenue", "scale": {"domain": [0, 1], "range": ["#636EFA", "#EF553B"]}},
            "tooltip": [{"field": "Temperature", "format": ".1f"}, {"field": "Revenue", "format": ",.0f"}]
        }
    }
    
    st.vega_lite_chart(data=sample, spec=spec, use_container_width=True) 

with right:
    st.subheader("🎯 Live Prediction Panel")
    st.markdown("Enter a **temperature** to see the model's prediction and confidence. Changes update instantly.")
    
    temp = st.number_input("Temperature (°C) 🌡️", value=float(df['Temperature'].median()), step=0.1, format="%.1f")

    x = np.array([[temp]])
    x_s = scaler.transform(x)
    pred = svm.predict(x_s)[0]
    proba = svm.predict_proba(x_s)[0]
    prob_high = float(proba[1])
    label = "HIGH REVENUE" if pred == 1 else "LOW REVENUE"

    st.write("")
    if pred == 1:
        st.success(f"### 🚀 Prediction: {label}")
    else:
        st.info(f"### ❄️ Prediction: {label}")

    p_low = float(proba[0])
    p_high = float(proba[1])
    
    st.markdown("#### Confidence Breakdown:")
    pcol1, pcol2 = st.columns(2)
    pcol1.metric(label="P(Low revenue)", value=f"{p_low:.1%}", delta_color="off")
    pcol2.metric(label="P(High revenue)", value=f"{p_high:.1%}", delta_color="off")
    
    st.progress(min(max(p_high, 0.0), 1.0), text=f"High Revenue Probability: {p_high:.1%}")

    st.markdown("---")
    st.markdown("#### 🗣️ Plain Language Interpretation:")
    if prob_high >= 0.8:
        st.success("The model is **highly confident** this temperature will produce **High revenue** compared to historical days. A great day for sales!")
    elif prob_high >= 0.6:
        st.warning("The model suggests it's **somewhat likely** — above average chance of High revenue.")
    elif prob_high >= 0.4:
        st.info("The model is **uncertain** — about a 50/50 chance. Pay attention to other factors.")
    else:
        st.error("The model suggests it's **likely Low revenue** for this temperature. Expect slower sales.")

    st.write("")
    st.subheader("⚙️ Quick Actions")
    
    model_path = os.path.join(MODEL_DIR, 'svm_model.joblib')
    try:
        with open(model_path, 'rb') as fh:
            model_bytes = fh.read()
        st.download_button('Download SVM Model (.joblib)', data=model_bytes, file_name='svm_model.joblib', mime='application/octet-stream')
    except Exception as e:
        st.error(f"Error loading model file for download: {e}")

st.markdown("---")
with st.expander("🛠️ Technical Details (Classification Reports & Confusion Matrices)"):
    st.subheader("Decision Tree Report")
    if 'decision_tree' in results:
        st.code(f"Confusion matrix:\n{results['decision_tree'].get('confusion_matrix')}")
        if isinstance(results['decision_tree'].get('classification_report'), dict):
            st.json(results['decision_tree']['classification_report'])

    st.subheader("SVM Report")
    if 'svm' in results:
        st.code(f"Confusion matrix:\n{results['svm'].get('confusion_matrix')}")
        if isinstance(results['svm'].get('classification_report'), dict):
            st.json(results['svm']['classification_report'])
    st.caption("Built from `training_notebook.ipynb` — models trained on the provided `Ice_Cream.csv` dataset.")

st.markdown(
    """
    <div style="text-align: center; margin-top: 30px; padding: 10px; border-top: 1px solid #ccc;">
        Developed by <b>Safoora</b> 
    </div>
    """,
    unsafe_allow_html=True,
)
