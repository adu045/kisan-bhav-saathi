"""Agricultural Price Predictor for Small Farmers  --  run: streamlit run app.py"""
import io
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import forecast as fc
from sample_data import generate

st.set_page_config(page_title="Kisan Bhav Saathi", page_icon="🌾", layout="wide")

# ----------------------------------------------------------------- languages
T = {
    "English": {
        "title": "🌾 Kisan Bhav Saathi – Crop Price Predictor",
        "sub": "Know the likely mandi price before you decide to sell.",
        "lang": "Language", "crop": "Crop", "market": "Market (Mandi)",
        "days": "Forecast days", "upload": "Upload real mandi CSV (optional)",
        "today": "Today's price", "best": "Best expected price (7 days)",
        "unit": "₹ / quintal", "sell": "SELL NOW", "hold": "HOLD",
        "unsure": "RISKY – WAIT AND WATCH",
        "sell_msg": "Prices are not expected to rise by much. Selling now is sensible.",
        "hold_msg": "Prices are expected to rise by about {g:.1f}% around {d}. Consider holding if you can store safely.",
        "unsure_msg": "A rise of {g:.1f}% is possible around {d}, but the forecast is uncertain. Decide based on storage cost and need for cash.",
        "chart": "Price history and forecast", "hist": "Past price",
        "pred": "Forecast", "range": "Likely range (80%)",
        "acc": "How reliable is this forecast?",
        "acc_txt": "In 5 past tests of 14-day forecasts, the model's average error was ₹{mae:.0f} per quintal ({mape:.1f}%). A simple 'tomorrow = today' guess had ₹{nmae:.0f} ({nmape:.1f}%).",
        "disc": "This is a guide, not a guarantee. Sudden weather, export rules or transport strikes can change prices.",
        "table": "Day-wise forecast", "date": "Date", "low": "Low", "high": "High",
        "sample": "Showing SAMPLE data. Upload real Agmarknet data for real prices.",
        "crops": {"Onion": "Onion", "Tomato": "Tomato", "Soybean": "Soybean"},
    },
    "मराठी": {
        "title": "🌾 किसान भाव साथी – पीक भाव अंदाज",
        "sub": "विकण्याआधी बाजारातील संभाव्य भाव जाणून घ्या.",
        "lang": "भाषा", "crop": "पीक", "market": "बाजार समिती",
        "days": "किती दिवसांचा अंदाज", "upload": "खरा बाजारभाव CSV अपलोड करा (ऐच्छिक)",
        "today": "आजचा भाव", "best": "पुढील ७ दिवसांतील सर्वोत्तम भाव",
        "unit": "₹ / क्विंटल", "sell": "आजच विका", "hold": "थांबा",
        "unsure": "जोखीम – थांबा आणि पाहा",
        "sell_msg": "भाव फारसे वाढण्याची शक्यता नाही. आता विकणे योग्य ठरेल.",
        "hold_msg": "{d} च्या सुमारास भाव सुमारे {g:.1f}% वाढण्याची शक्यता आहे. सुरक्षित साठवण शक्य असेल तर थांबा.",
        "unsure_msg": "{d} च्या सुमारास {g:.1f}% वाढ शक्य आहे, पण अंदाज अनिश्चित आहे. साठवणूक खर्च आणि पैशांची गरज पाहून निर्णय घ्या.",
        "chart": "भावाचा इतिहास आणि अंदाज", "hist": "मागील भाव",
        "pred": "अंदाज", "range": "संभाव्य श्रेणी (८०%)",
        "acc": "हा अंदाज किती विश्वासार्ह आहे?",
        "acc_txt": "१४ दिवसांच्या ५ मागील चाचण्यांत मॉडेलची सरासरी चूक ₹{mae:.0f} प्रति क्विंटल ({mape:.1f}%) होती. 'उद्याचा भाव = आजचा भाव' या साध्या अंदाजात ती ₹{nmae:.0f} ({nmape:.1f}%) होती.",
        "disc": "हा फक्त मार्गदर्शक अंदाज आहे, हमी नाही. हवामान, निर्यात नियम किंवा संप यामुळे भाव बदलू शकतात.",
        "table": "दिवसनिहाय अंदाज", "date": "तारीख", "low": "किमान", "high": "कमाल",
        "sample": "नमुना (SAMPLE) डेटा दाखवला आहे. खऱ्या भावासाठी Agmarknet डेटा अपलोड करा.",
        "crops": {"Onion": "कांदा", "Tomato": "टोमॅटो", "Soybean": "सोयाबीन"},
    },
    "हिन्दी": {
        "title": "🌾 किसान भाव साथी – फसल भाव पूर्वानुमान",
        "sub": "बेचने से पहले मंडी का संभावित भाव जानें।",
        "lang": "भाषा", "crop": "फसल", "market": "मंडी",
        "days": "कितने दिन का पूर्वानुमान", "upload": "असली मंडी भाव CSV अपलोड करें (वैकल्पिक)",
        "today": "आज का भाव", "best": "अगले 7 दिनों का सबसे अच्छा भाव",
        "unit": "₹ / क्विंटल", "sell": "अभी बेचें", "hold": "रुकें",
        "unsure": "जोखिम – रुकें और देखें",
        "sell_msg": "भाव ज़्यादा बढ़ने की उम्मीद नहीं है। अभी बेचना ठीक रहेगा।",
        "hold_msg": "{d} के आसपास भाव लगभग {g:.1f}% बढ़ने की संभावना है। सुरक्षित भंडारण हो तो रुकें।",
        "unsure_msg": "{d} के आसपास {g:.1f}% बढ़त संभव है, पर अनुमान अनिश्चित है। भंडारण खर्च और नकदी की ज़रूरत देखकर फैसला करें।",
        "chart": "भाव का इतिहास और पूर्वानुमान", "hist": "पिछला भाव",
        "pred": "पूर्वानुमान", "range": "संभावित दायरा (80%)",
        "acc": "यह पूर्वानुमान कितना भरोसेमंद है?",
        "acc_txt": "14 दिन के 5 पिछले परीक्षणों में मॉडल की औसत गलती ₹{mae:.0f} प्रति क्विंटल ({mape:.1f}%) रही। 'कल = आज' वाले सरल अनुमान में यह ₹{nmae:.0f} ({nmape:.1f}%) थी।",
        "disc": "यह केवल मार्गदर्शन है, गारंटी नहीं। मौसम, निर्यात नियम या हड़ताल से भाव बदल सकते हैं।",
        "table": "दिन-वार पूर्वानुमान", "date": "तारीख", "low": "न्यूनतम", "high": "अधिकतम",
        "sample": "नमूना (SAMPLE) डेटा दिखाया जा रहा है। असली भाव के लिए Agmarknet डेटा अपलोड करें।",
        "crops": {"Onion": "प्याज़", "Tomato": "टमाटर", "Soybean": "सोयाबीन"},
    },
}

# ---------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load_default() -> pd.DataFrame:
    if not os.path.exists("mandi_prices.csv"):
        generate().to_csv("mandi_prices.csv", index=False)
    return pd.read_csv("mandi_prices.csv", parse_dates=["date"])


@st.cache_data(show_spinner="Training model…")
def predict(df_json: str, horizon: int):
    series = fc.clean_series(pd.read_json(io.StringIO(df_json)))
    return fc.run(series, horizon=horizon)


# ------------------------------------------------------------------- sidebar
lang = st.sidebar.selectbox("Language / भाषा", list(T.keys()))
t = T[lang]

uploaded = st.sidebar.file_uploader(t["upload"], type="csv")
if uploaded:
    data = pd.read_csv(uploaded, parse_dates=["date"])
    is_sample = False
else:
    data = load_default()
    is_sample = True

crops = sorted(data["crop"].unique())
crop = st.sidebar.selectbox(t["crop"], crops, format_func=lambda c: t["crops"].get(c, c))
markets = sorted(data.loc[data["crop"] == crop, "market"].unique())
market = st.sidebar.selectbox(t["market"], markets)
horizon = st.sidebar.slider(t["days"], 7, 21, 14)

# ---------------------------------------------------------------------- main
st.title(t["title"])
st.caption(t["sub"])
if is_sample:
    st.info(t["sample"])

sel = data[(data["crop"] == crop) & (data["market"] == market)][["date", "modal_price"]]
if len(sel) < 90:
    st.error("Need at least 90 days of data for this crop and market.")
    st.stop()

res = predict(sel.to_json(date_format="iso"), horizon)
current = float(res.history.iloc[-1])
decision, day, gain = fc.advise(current, res.forecast)
day_txt = day.strftime("%d %b")

c1, c2 = st.columns(2)
c1.metric(f"{t['today']} ({t['unit']})", f"₹{current:,.0f}")
best7 = res.forecast["price"].iloc[:7].max()
c2.metric(f"{t['best']} ({t['unit']})", f"₹{best7:,.0f}", f"{(best7 - current) / current * 100:+.1f}%")

box = {"SELL": (st.error, "sell"), "HOLD": (st.success, "hold"), "UNSURE": (st.warning, "unsure")}
fn, key = box[decision]
fn(f"### {t[key]}\n{t[key + '_msg'].format(g=abs(gain), d=day_txt)}")

# chart
hist = res.history.iloc[-120:]
f = res.forecast
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name=t["hist"], line=dict(color="#2e7d32", width=2)))
fig.add_trace(go.Scatter(x=list(f.index) + list(f.index[::-1]), y=list(f["high"]) + list(f["low"][::-1]),
                         fill="toself", fillcolor="rgba(255,152,0,0.2)", line=dict(width=0),
                         name=t["range"], hoverinfo="skip"))
fig.add_trace(go.Scatter(x=f.index, y=f["price"], name=t["pred"], line=dict(color="#ef6c00", width=3, dash="dash")))
fig.update_layout(title=t["chart"], yaxis_title=t["unit"], height=420, margin=dict(t=50, b=10),
                  legend=dict(orientation="h", y=-0.15))
st.plotly_chart(fig, width="stretch")

with st.expander(t["acc"]):
    st.write(t["acc_txt"].format(mae=res.mae, mape=res.mape, nmae=res.naive_mae, nmape=res.naive_mape))
    st.caption(t["disc"])

with st.expander(t["table"]):
    tbl = f.rename(columns={"price": t["pred"], "low": t["low"], "high": t["high"]})
    tbl.index = tbl.index.strftime("%d %b %Y")
    tbl.index.name = t["date"]
    st.dataframe(tbl, width="stretch")

st.caption(t["disc"])