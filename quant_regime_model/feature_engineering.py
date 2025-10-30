import numpy as np
import pandas as pd
import ssm
from arch import arch_model
from colorama import Fore, Style

def create_price_action_features(open_p, high_p, low_p, close_p):
    body_size = (close_p - open_p).abs()
    upper_wick = high_p - pd.concat([open_p, close_p], axis=1).max(axis=1)
    lower_wick = pd.concat([open_p, close_p], axis=1).min(axis=1) - low_p
    total_range = high_p - low_p + 1e-6 # Avoid zero division
    
    position_in_range = (close_p - low_p) / total_range
    gap = open_p - close_p.shift(1)
    
    return pd.DataFrame({
        'body_size': body_size,
        'upper_wick': upper_wick,
        'lower_wick': lower_wick,
        'position_in_range': position_in_range,
        'gap': gap
    })

def create_momentum_and_zscore(returns, mom_windows, zscore_window):
    features = {}
    for window in mom_windows:
        features[f'momentum_{window}'] = returns.rolling(window).mean()
        
    z_score = (returns - returns.rolling(zscore_window).mean()) / returns.rolling(zscore_window).std()
    features[f'zscore_{zscore_window}'] = z_score
    return pd.DataFrame(features)

def fit_hmm_and_get_features(returns, k_max, alpha, kappa, n_iters):
    print(f"{Fore.YELLOW}Fitting Sticky HDP-HMM...{Style.RESET_ALL}")
    
    hmm = ssm.HMM(K=k_max, D=1, observations="gaussian",
                  transitions="sticky",
                  transition_kwargs={"alpha": alpha, "kappa": kappa})
    
    hmm.fit(returns.reshape(-1, 1), method="em", n_iters=n_iters, tolerance=1e-4)
    
    regimes = hmm.most_likely_states(returns.reshape(-1, 1))
    regime_probs = hmm.expected_states(returns.reshape(-1, 1))[0]
    
    used_states = np.unique(regimes)
    K_found = len(used_states)
    
    state_map = {old: new for new, old in enumerate(used_states)}
    regimes = np.array([state_map[r] for r in regimes])
    regime_probs = regime_probs[:, used_states]
    
    print(f"{Fore.GREEN}HMM fit complete. Found {K_found} regimes.{Style.RESET_ALL}")
    
    prob_cols = {f'regime_prob_{i}': regime_probs[:, i] for i in range(K_found)}
    
    return pd.DataFrame(prob_cols, index=returns.index), regimes, K_found

def get_regime_garch_vol(returns, regimes, K_found):
    print("Calculating MS-GARCH features...")
    garch_forecasts = []
    
    for r in range(K_found):
        regime_returns = returns[regimes == r] * 100
        if len(regime_returns) < 30:
            garch_forecasts.append(np.nan)
            continue
        try:
            garch = arch_model(regime_returns, vol='Garch', p=1, q=1, mean='Zero')
            fitted = garch.fit(disp='off')
            forecast = fitted.forecast(horizon=1).variance.iloc[-1].values[0] / 100
            garch_forecasts.append(forecast)
        except:
            garch_forecasts.append(np.nan)

    garch_forecasts = pd.Series(garch_forecasts).fillna(method='ffill').fillna(method='bfill').values
    vol_forecast = np.array([garch_forecasts[r] for r in regimes])
    
    return pd.Series(vol_forecast, index=returns.index, name='regime_vol_forecast')

def build_features(data, ticker_name, mom_windows, zscore_window, hmm_params):
    
    print(f"Building features for {ticker_name}...")
    
    close = data[('Close', ticker_name)]
    open_p = data[('Open', ticker_name)]
    high_p = data[('High', ticker_name)]
    low_p = data[('Low', ticker_name)]
    
    returns = close.pct_change().dropna()
    
    pa_features = create_price_action_features(open_p, high_p, low_p, close)
    mom_features = create_momentum_and_zscore(returns, mom_windows, zscore_window)
    
    hmm_features, regimes, K_found = fit_hmm_and_get_features(
        returns.values, **hmm_params
    )
    
    garch_features = get_regime_garch_vol(returns, regimes, K_found)
    
    all_features = pd.concat([returns.rename('returns'), pa_features, mom_features, hmm_features, garch_features], axis=1)
    
    print("Feature building complete.")
    return all_features, returns