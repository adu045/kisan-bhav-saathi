# Kisan Bhav Saathi – Agricultural Price Predictor for Small Farmers

A Streamlit web app that forecasts mandi prices for the next 7–21 days and tells a
farmer in simple Marathi / Hindi / English whether to **sell now** or **hold**.

## Run

```bash
pip install -r requirements.txt
python sample_data.py        # creates SAMPLE mandi_prices.csv (first run does this automatically)
streamlit run app.py
```

## Use REAL data (do this before the presentation)

1. Download daily mandi prices from https://agmarknet.gov.in or https://data.gov.in
   (search "Current daily price of various commodities from various markets").
2. Save a CSV with exactly these columns, one row per day per market per crop:
   `date, market, crop, modal_price`   (modal_price in Rs/quintal)
3. Either replace `mandi_prices.csv`, or upload the file from the app sidebar.
   Need at least ~6 months of daily data per crop/market.

## How it works

| File | Role |
|---|---|
| `sample_data.py` | Generates seasonal, mean-reverting sample prices (for demo only) |
| `forecast.py` | Cleans data, builds features, trains model, forecasts, backtests, gives advice |
| `app.py` | Streamlit UI, 3 languages, chart, sell/hold box, reliability section |

**Model:** Ridge regression predicts the next-day change in log-price from
(a) the 7-day price change, (b) distance from the 30-day average (mean reversion),
(c) yearly seasonality (sin/cos of day of year). It is applied recursively for
the forecast horizon, with each daily step capped at 2 standard deviations.

**Uncertainty:** the orange band is an ~80% range from the model's residual
spread, growing with the square root of the days ahead.

**Honest accuracy:** the app backtests the model on 5 past start dates (14-day
forecasts, trained only on earlier data) and compares it with the naive guess
"tomorrow = today". On the sample data the model is only slightly better than the
naive guess – that is expected for sample data. Report the numbers you get on real data.

**Advice rule (`advise()` in forecast.py):**
- HOLD if the best price in the next 7 days is >= 4% above today AND the low end of the range is not far below today
- UNSURE if a rise is forecast but the range is wide
- SELL otherwise

## Viva cheat-sheet

- *Why Ridge, not LSTM/Prophet?* Small data, needs to be fast and explainable; a heavy model would overfit.
- *Why predict log-returns?* Makes prices comparable across crops/markets and keeps forecasts positive.
- *Limits?* Cannot see weather, export bans, strikes, or arrivals volume. Next step: add rainfall and arrival-quantity features.
- *Why a range?* A single number would give false confidence.
- *Farmer without a smartphone?* Next step: SMS/WhatsApp bot using the same `forecast.py`.