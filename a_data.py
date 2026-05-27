import pandas as pd
import numpy as np
import yfinance as yf
import os

base = os.path.dirname(__file__)
TgP = pd.read_csv(os.path.join(base, "TgP_weights.csv"), index_col=0)
weights = TgP.iloc[:,0]
weights = weights[weights>0]
tickers = list(weights.index)
start = "2016-04-30"
end = "2026-04-30"

def get_returns(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Funzione che ritorna un df di log-rendimenti dei ticker di interesse nella finestra temporale richiesta"""
    data = yf.download(tickers, start= start, end= end)["Close"]
    rend = pd.DataFrame(np.log(data/data.shift(1)).dropna())
    return rend