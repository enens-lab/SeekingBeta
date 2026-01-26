# Quick diagnostics to test data sources without the full training run.
from data.fetch import fetch_ohlcv

print("Testing Alpaca...")
try:
    d = fetch_ohlcv("AAPL", period="30d", data_source="alpaca")
    print("Alpaca OK:\n", d.tail())
except Exception as e:
    print("Alpaca failed:", e)

print("\nTesting Yahoo...")
try:
    d = fetch_ohlcv("AAPL", period="5d", data_source="yahoo")
    print("Yahoo OK:\n", d.tail())
except Exception as e:
    print("Yahoo failed:", e)

print("\nTesting Stooq...")
try:
    d = fetch_ohlcv("AAPL", period="5d", data_source="stooq")
    print("Stooq OK:\n", d.tail())
except Exception as e:
    print("Stooq failed:", e)
