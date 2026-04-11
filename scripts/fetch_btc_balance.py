"""
Fetches BTC balance from Binance and writes it to portfolio.json.
Run by GitHub Actions every 30 minutes.
"""
import json
import os
from datetime import datetime, timezone

from binance.client import Client


def main():
    api_key = os.environ["BINANCE_API_KEY"]
    api_secret = os.environ["BINANCE_API_SECRET"]

    client = Client(api_key, api_secret)
    info = client.get_asset_balance(asset="BTC")
    btc_balance = float(info["free"]) + float(info["locked"])

    with open("portfolio.json", "r") as f:
        portfolio = json.load(f)

    portfolio["btc_balance"] = btc_balance
    portfolio["btc_balance_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    with open("portfolio.json", "w") as f:
        json.dump(portfolio, f, indent=2)

    print(f"BTC balance updated: {btc_balance:.8f} BTC")


if __name__ == "__main__":
    main()
