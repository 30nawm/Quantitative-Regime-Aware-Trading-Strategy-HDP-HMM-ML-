import datetime

# --- Ticker and Data Settings ---
TICKERS = {
    "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ Composite",
    "^FTSE": "FTSE 100"
}
END_DATE = datetime.date.today()
START_DATE = END_DATE - datetime.timedelta(days=5*365) # 5 years of data

# --- Feature Engineering Settings ---
MOMENTUM_WINDOWS = [10, 20, 60]
ZSCORE_WINDOW = 20

# --- Regime Model (HDP-HMM) Settings ---
HMM_K_MAX = 15      # Max regimes to find
HMM_ALPHA = 1.0     # Low: prefer fewer new regimes
HMM_KAPPA = 100.0   # High: prefer sticky (persistent) regimes
HMM_N_ITERS = 100

# --- Labeling (Triple Barrier Method) Settings ---
# Note: Window is in days (trading days)
TBM_LOOKAHEAD_WINDOW = 10  # How far to look for barrier touch
TBM_PROFIT_BARRIER = 0.03  # 3% take profit
TBM_LOSS_BARRIER = 0.02    # 2% stop loss

# --- Model Training Settings ---
TEST_SPLIT_SIZE = 0.2 # 20% of data for testing
NN_EPOCHS = 50
NN_BATCH_SIZE = 32

# --- Backtesting and Signal Settings ---
# High-conviction thresholds as requested
SIGNAL_BUY_THRESHOLD = 0.9   # P(Buy) > 90%
SIGNAL_SELL_THRESHOLD = 0.9  # P(Sell) > 90%
RISK_FREE_RATE = 0.01 # Annualized risk-free rate for Sharpe