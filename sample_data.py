"""Generate realistic SAMPLE mandi price data so the app runs out of the box.

Replace mandi_prices.csv with real Agmarknet / data.gov.in data for your final
submission (columns: date, market, crop, modal_price in Rs/quintal).
Run:  python sample_data.py
"""
import numpy as np
import pandas as pd

CROPS = {
    # crop: (base price Rs/quintal, seasonal amplitude, peak day-of-year, volatility)
    "Onion": (1800, 0.35, 330, 0.035),
    "Tomato": (1500, 0.40, 150, 0.060),
    "Soybean": (4500, 0.10, 300, 0.012),
}
MARKETS = {"Nashik": 1.00, "Pune": 1.06, "Nagpur": 0.97}


def generate(days: int = 900, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    dates = pd.date_range(end=end, periods=days, freq="D")
    rows = []
    for crop, (base, amp, peak, vol) in CROPS.items():
        # shared national trend + seasonality
        doy = dates.dayofyear.values
        season = amp * np.cos(2 * np.pi * (doy - peak) / 365.25)
        shock = np.zeros(days)
        for i in range(1, days):                       # mean-reverting (AR(1)) shocks
            shock[i] = 0.985 * shock[i - 1] + rng.normal(0, vol * 0.6)
        national = np.exp(season + shock)
        for market, factor in MARKETS.items():
            local = np.zeros(days)
            for i in range(1, days):
                local[i] = 0.9 * local[i - 1] + rng.normal(0, vol * 0.3)
            price = base * factor * national * np.exp(local)
            price = np.clip(price, base * 0.25, None).round(0)
            rows.append(pd.DataFrame({"date": dates, "market": market,
                                      "crop": crop, "modal_price": price}))
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    df = generate()
    df.to_csv("mandi_prices.csv", index=False)
    print(f"Saved mandi_prices.csv with {len(df)} rows")