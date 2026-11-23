"""Download real market candles for Rohit's morning brief.

Runs on GitHub Actions every weekday morning and writes:
  data/summary.csv          one row per symbol: last close, previous high/low,
                            20-day average and its slope, % moves
  data/daily/<name>.csv     about 3 months of daily candles
  data/intraday/<name>.csv  5-minute candles for the last 5 sessions (IST)
  data/options/sensex_<expiry>.json  BSE Sensex option chain (if BSE allows it)
  data/meta.json            when it ran and which symbols failed
"""

import json
import os
import time
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests
import yfinance as yf

IST = timezone(timedelta(hours=5, minutes=30))
OUT = "data"

INDICES = {
    "NIFTY50": "^NSEI", "SENSEX": "^BSESN", "BANKNIFTY": "^NSEBANK",
    "INDIAVIX": "^INDIAVIX", "NIFTY_IT": "^CNXIT", "NIFTY_AUTO": "^CNXAUTO",
    "NIFTY_METAL": "^CNXMETAL", "NIFTY_PHARMA": "^CNXPHARMA",
    "NIFTY_FMCG": "^CNXFMCG", "NIFTY_PSUBANK": "^CNXPSUBANK",
    "NIFTY_REALTY": "^CNXREALTY", "NIFTY_ENERGY": "^CNXENERGY",
    "NIFTY_MEDIA": "^CNXMEDIA", "NIFTY_FINSERV": "NIFTY_FIN_SERVICE.NS",
}

GLOBAL = {
    "SP500": "^GSPC", "NASDAQ": "^IXIC", "DOW": "^DJI", "FTSE100": "^FTSE",
    "DAX": "^GDAXI", "CAC40": "^FCHI", "NIKKEI": "^N225", "HANGSENG": "^HSI",
    "SHANGHAI": "000001.SS", "KOSPI": "^KS11", "BRENT": "BZ=F",
    "USDINR": "INR=X", "US10Y": "^TNX", "DXY": "DX-Y.NYB", "GOLD": "GC=F",
}

# Liquid NSE F&O stocks. Add or remove symbols here (NSE symbol, no .NS).
STOCKS = """
RELIANCE HDFCBANK ICICIBANK INFY TCS BHARTIARTL SBIN ITC LT KOTAKBANK AXISBANK
HINDUNILVR BAJFINANCE MARUTI M&M SUNPHARMA HCLTECH TITAN ULTRACEMCO NTPC
TATASTEEL POWERGRID ONGC ASIANPAINT ADANIENT ADANIPORTS COALINDIA BAJAJFINSV
NESTLEIND JSWSTEEL GRASIM HINDALCO WIPRO TECHM CIPLA DRREDDY EICHERMOT
HEROMOTOCO BAJAJ-AUTO BRITANNIA APOLLOHOSP TATACONSUM SBILIFE HDFCLIFE
INDUSINDBK SHRIRAMFIN BEL TRENT ETERNAL JIOFIN BANKBARODA PNB CANBK
UNIONBANK IDFCFIRSTB FEDERALBNK AUBANK BANDHANBNK HAL BHEL IRFC RECLTD
PFC TATAPOWER ADANIGREEN ADANIPOWER GAIL IOC BPCL HPCL VEDL NMDC SAIL
JINDALSTEL HINDZINC DLF GODREJPROP LODHA OBEROIRLTY DIVISLAB LUPIN
AUROPHARMA ZYDUSLIFE TORNTPHARM MANKIND ALKEM BIOCON MAXHEALTH DMART
GODREJCP DABUR MARICO COLPAL VBL UNITDSPR TVSMOTOR ASHOKLEY BOSCHLTD
MOTHERSON BHARATFORG TATAMOTORS LTIM PERSISTENT COFORGE MPHASIS OFSS
KPITTECH TATAELXSI POLYCAB HAVELLS DIXON SIEMENS ABB CGPOWER CUMMINSIND
PIDILITIND BERGEPAINT SRF PIIND UPL CHOLAFIN MUTHOOTFIN MANAPPURAM
BAJAJHLDNG LICI LICHSGFIN PAYTM POLICYBZR NYKAA IRCTC INDIGO INDHOTEL
BSE CDSL MCX ANGELONE KAYNES APLAPOLLO AMBUJACEM ACC SHREECEM DALBHARAT
NAUKRI PAGEIND KALYANKJIL CROMPTON VOLTAS BLUESTARCO
""".split()


def ist_index(df):
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert(IST)
    return df


def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]]


def fetch(name, ticker, failures):
    row = {"name": name, "ticker": ticker}
    try:
        daily = yf.download(ticker, period="4mo", interval="1d",
                            progress=False, auto_adjust=False, threads=False)
        daily = flatten(daily).dropna(subset=["Close"])
        if daily.empty:
            raise ValueError("no daily data")
        daily.index = pd.to_datetime(daily.index).date
        daily.round(2).to_csv(f"{OUT}/daily/{name}.csv", index_label="Date")

        closes = daily["Close"]
        last = daily.iloc[-1]
        prev = daily.iloc[-2] if len(daily) > 1 else last
        sma20 = closes.rolling(20).mean()
        row.update({
            "last_date": str(daily.index[-1]),
            "close": round(last["Close"], 2),
            "high": round(last["High"], 2),
            "low": round(last["Low"], 2),
            "prev_close": round(prev["Close"], 2),
            "pct_change": round((last["Close"] / prev["Close"] - 1) * 100, 2),
            "pct_5d": round((closes.iloc[-1] / closes.iloc[-6] - 1) * 100, 2) if len(closes) > 5 else None,
            "sma20": round(sma20.iloc[-1], 2) if len(closes) >= 20 else None,
            "sma20_5d_ago": round(sma20.iloc[-6], 2) if len(closes) >= 25 else None,
            "above_sma20": bool(last["Close"] > sma20.iloc[-1]) if len(closes) >= 20 else None,
            "high_10d": round(daily["High"].iloc[-10:].max(), 2),
            "low_10d": round(daily["Low"].iloc[-10:].min(), 2),
            "high_20d": round(daily["High"].iloc[-20:].max(), 2),
            "low_20d": round(daily["Low"].iloc[-20:].min(), 2),
        })
        if row["sma20"] and row["sma20_5d_ago"]:
            row["sma20_rising"] = row["sma20"] > row["sma20_5d_ago"]

        intra = yf.download(ticker, period="5d", interval="5m",
                            progress=False, auto_adjust=False, threads=False)
        intra = flatten(intra).dropna(subset=["Close"])
        if not intra.empty:
            intra = ist_index(intra)
            intra.round(2).to_csv(f"{OUT}/intraday/{name}.csv", index_label="Datetime_IST")
            last_day = intra[intra.index.date == intra.index[-1].date()]
            vol = last_day["Volume"].fillna(0)
            typical = (last_day["High"] + last_day["Low"] + last_day["Close"]) / 3
            weights = vol if vol.sum() > 0 else pd.Series(1, index=last_day.index)
            row["intraday_last_date"] = str(intra.index[-1].date())
            row["vwap_last_session"] = round((typical * weights).sum() / weights.sum(), 2)
            row["vwap_is_time_weighted"] = bool(vol.sum() == 0)
    except Exception as e:  # keep going; record the failure
        failures.append(f"{name} ({ticker}): {e}")
    return row


def bse_sensex_chain(failures):
    """Save the nearest Sensex weekly option chain from BSE, if reachable."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Referer": "https://www.bseindia.com/",
        "Origin": "https://www.bseindia.com",
        "Accept": "application/json, text/plain, */*",
    })
    try:
        exp = s.get("https://api.bseindia.com/BseIndiaAPI/api/ddlExpiry/w"
                    "?ProductType=IO&scrip_cd=1", timeout=20).json()
        expiry = exp["Table"][0]["eXPIRY"]  # e.g. "26 Nov 2026"
        chain = s.get("https://api.bseindia.com/BseIndiaAPI/api/DerivOptionChain_IV/w",
                      params={"Expiry": expiry, "scrip_cd": 1, "strprice": 0},
                      timeout=30).json()
        fname = expiry.replace(" ", "_")
        with open(f"{OUT}/options/sensex_{fname}.json", "w") as f:
            json.dump(chain, f)
        with open(f"{OUT}/options/sensex_latest.json", "w") as f:
            json.dump({"expiry": expiry, "saved_ist": datetime.now(IST).isoformat(),
                       "chain": chain}, f)
    except Exception as e:
        failures.append(f"BSE Sensex option chain: {e}")


def main():
    for d in ("daily", "intraday", "options"):
        os.makedirs(f"{OUT}/{d}", exist_ok=True)
    failures, rows = [], []
    universe = {**INDICES, **GLOBAL, **{s: f"{s}.NS" for s in STOCKS}}
    for name, ticker in universe.items():
        rows.append(fetch(name, ticker, failures))
        time.sleep(0.3)  # be gentle with Yahoo
    pd.DataFrame(rows).to_csv(f"{OUT}/summary.csv", index=False)
    bse_sensex_chain(failures)
    with open(f"{OUT}/meta.json", "w") as f:
        json.dump({"run_ist": datetime.now(IST).isoformat(timespec="seconds"),
                   "symbols": len(universe), "failures": failures}, f, indent=1)
    ok = sum(1 for r in rows if "close" in r)
    print(f"Done: {ok} of {len(universe)} symbols ok, {len(failures)} problems")
    for x in failures:
        print("FAILED", x)


if __name__ == "__main__":
    main()
