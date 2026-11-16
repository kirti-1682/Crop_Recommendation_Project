# 🌾 Crop Recommendation Web App (Streamlit) - interactive, colourful version
# Run:  streamlit run app.py      (from inside the app/ folder)
#   or: streamlit run app/app.py  (from the project root)

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
DATA_FILE = ROOT / "data" / "Crop_recommendation.csv"

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

EMOJI = {
    "apple": "🍎", "banana": "🍌", "blackgram": "🫘", "chickpea": "🫘", "coconut": "🥥",
    "coffee": "☕", "cotton": "☁️", "grapes": "🍇", "jute": "🧵", "kidneybeans": "🫘",
    "lentil": "🫘", "maize": "🌽", "mango": "🥭", "mothbeans": "🫘", "mungbean": "🫘",
    "muskmelon": "🍈", "orange": "🍊", "papaya": "🥭", "pigeonpeas": "🫛",
    "pomegranate": "🍎", "rice": "🌾", "watermelon": "🍉",
}

# ---------- Page setup ----------
st.set_page_config(page_title="Crop Recommendation System", page_icon="🌾", layout="wide")

st.markdown(
    """
    <style>
    /* hide the Deploy button */
    .stAppDeployButton,
    div[data-testid="stAppDeployButton"] { display: none !important; }

    /* push content to the very top */
    .block-container { padding-top: 0.5rem !important; }
    header[data-testid="stHeader"] { background: transparent; height: 2.5rem; }

    /* green bar: flush at the top, name at top-left */
    .hero {
        background: linear-gradient(120deg, #1b5e20 0%, #2e9e6b 50%, #1976d2 100%);
        padding: 18px 28px 22px; border-radius: 16px; color: white;
        margin: 0 0 18px 0; text-align: left;
    }
    .hero h1 { margin: 0 0 6px 0; padding: 0; color: white; font-size: 2rem; text-align: left; }
    .hero p  { margin: 0; font-size: 1.05rem; opacity: .95; }

    .result-card {
        background: linear-gradient(135deg, #e8f5e9, #bbdefb);
        border-left: 8px solid #1976d2; border-radius: 14px; padding: 22px 26px; color: #0d2b3e;
    }
    .result-card .label { font-size: .95rem; opacity: .75; }
    .result-card .crop  { font-size: 2.8rem; font-weight: 800; line-height: 1.1; }
    .result-card .conf  { font-size: 1.05rem; font-weight: 600; }
    .bar-row { margin: 10px 0; }
    .bar-top { display: flex; justify-content: space-between; font-weight: 600; margin-bottom: 3px; }
    .bar-bg  { background: #e3f2fd; border-radius: 8px; height: 14px; overflow: hidden; }
    .bar-fg  { height: 14px; border-radius: 8px; }
    section[data-testid="stSidebar"] { background: #e3f2fd; }
    div[data-testid="stMetric"] {
        background: #e8f5e9; border-radius: 12px; padding: 10px 14px; border: 1px solid #90caf9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Load model, scaler, encoder and dataset (cached) ----------
@st.cache_resource
def load_objects():
    def load(name):
        with open(MODEL_DIR / name, "rb") as f:
            return pickle.load(f)
    return load("crop_model.pkl"), load("scaler.pkl"), load("label_encoder.pkl")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    ranges = {c: (float(df[c].min()), float(df[c].max()), float(df[c].median())) for c in FEATURES}
    typical = df.groupby("label")[FEATURES].median().round(1)
    return ranges, typical


model, scaler, label_encoder = load_objects()
ranges, typical = load_data()


def rng(col, step):
    lo, hi, med = ranges[col]
    return float(lo), float(hi), float(round(med / step) * step)


# ---------- Session state: slider defaults + "typical crop" loader ----------
DEFAULTS = {
    "N": 90, "P": 42, "K": 43, "ph": 6.5,
    "temperature": 25.5, "humidity": 80.0, "rainfall": 200.0,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def load_example():
    crop = st.session_state["example"]
    if crop != "(none)":
        for col in FEATURES:
            val = float(typical.loc[crop, col])
            st.session_state[col] = int(val) if col in ("N", "P", "K") else val


# ---------- Sidebar: inputs ----------
with st.sidebar:
    st.header("🧪 Your field")
    st.selectbox(
        "Try typical conditions for…",
        ["(none)"] + sorted(typical.index),
        key="example",
        on_change=load_example,
        format_func=lambda c: c if c == "(none)" else f"{EMOJI.get(c, '🌱')} {c.title()}",
    )

    st.subheader("🟤 Soil")
    N = st.slider("Nitrogen (N)", int(ranges["N"][0]), int(ranges["N"][1]), key="N")
    P = st.slider("Phosphorus (P)", int(ranges["P"][0]), int(ranges["P"][1]), key="P")
    K = st.slider("Potassium (K)", int(ranges["K"][0]), int(ranges["K"][1]), key="K")
    ph = st.slider("Soil pH", 3.5, 10.0, step=0.1, key="ph")

    st.subheader("🌦️ Weather")
    temperature = st.slider("Temperature (°C)", 8.0, 44.0, step=0.1, key="temperature")
    humidity = st.slider("Humidity (%)", 14.0, 100.0, step=0.5, key="humidity")
    rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, step=1.0, key="rainfall")

# ---------- Prediction (runs on every slider move) ----------
values = {"N": N, "P": P, "K": K, "temperature": temperature,
          "humidity": humidity, "ph": ph, "rainfall": rainfall}
row = pd.DataFrame([values], columns=FEATURES)
scaled = pd.DataFrame(scaler.transform(row), columns=FEATURES)
probs = model.predict_proba(scaled)[0]
order = probs.argsort()[::-1][:5]
names = label_encoder.inverse_transform(model.classes_[order])
top = [(str(n), float(probs[i]) * 100) for n, i in zip(names, order)]
best, best_p = top[0]

# ---------- Main page ----------
st.markdown(
    """<div class="hero"><h1>🌾 Crop Recommendation System</h1>
    <p>Move the sliders on the left. The best crop for your field updates instantly.</p></div>""",
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1.1], gap="large")

with left:
    st.markdown(
        f"""<div class="result-card">
            <div class="label">Best match for your field</div>
            <div class="crop">{EMOJI.get(best, '🌱')} {best.title()}</div>
            <div class="conf">{best_p:.0f}% confidence</div>
        </div>""",
        unsafe_allow_html=True,
    )
    if best_p < 50:
        st.info("Several crops fit these conditions. Check the runners-up on the right.")

    st.markdown("#### Your readings")
    c1, c2, c3 = st.columns(3)
    c1.metric("Nitrogen", N); c2.metric("Phosphorus", P); c3.metric("Potassium", K)
    c4, c5, c6 = st.columns(3)
    c4.metric("pH", ph); c5.metric("Temp (°C)", temperature); c6.metric("Humidity (%)", humidity)
    st.metric("Rainfall (mm)", rainfall)

with right:
    st.markdown("#### Top 5 matches")
    palette = ["#2e7d32", "#1e88e5", "#66bb6a", "#42a5f5", "#80cbc4"]
    bars = ""
    for (name, p), colour in zip(top, palette):
        bars += (
            f'<div class="bar-row"><div class="bar-top"><span>{EMOJI.get(name, "🌱")} {name.title()}</span>'
            f'<span>{p:.0f}%</span></div><div class="bar-bg">'
            f'<div class="bar-fg" style="width:{max(p, 1):.0f}%;background:{colour}"></div></div></div>'
        )
    st.markdown(bars, unsafe_allow_html=True)

    st.markdown(f"#### Your field vs typical {best.title()}")
    compare = pd.DataFrame({"Your field": row.iloc[0], f"Typical {best.title()}": typical.loc[best]})
    st.dataframe(compare.round(1))

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit | Crop Recommendation ML Model")