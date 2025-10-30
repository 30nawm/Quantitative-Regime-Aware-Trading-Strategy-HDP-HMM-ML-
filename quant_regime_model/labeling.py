import numpy as np
import pandas as pd

def get_triple_barrier_labels(close_prices, window, pt_barrier, sl_barrier):
    
    n_days = len(close_prices)
    labels = pd.Series(np.zeros(n_days), index=close_prices.index)
    
    for i in range(n_days - window):
        entry_price = close_prices.iloc[i]
        
        take_profit_price = entry_price * (1 + pt_barrier)
        stop_loss_price = entry_price * (1 - sl_barrier)
        
        future_path = close_prices.iloc[i+1 : i+1+window]
        
        # Check for barrier touches
        touch_tp = (future_path >= take_profit_price).any()
        touch_sl = (future_path <= stop_loss_price).any()
        
        if touch_tp and touch_sl:
            # If both hit, see which one hit first
            tp_first = future_path[future_path >= take_profit_price].index[0]
            sl_first = future_path[future_path <= stop_loss_price].index[0]
            if tp_first <= sl_first:
                labels.iloc[i] = 1  # Buy (Profit)
            else:
                labels.iloc[i] = -1 # Sell (Loss)
                
        elif touch_tp:
            labels.iloc[i] = 1 # Buy (Profit)
            
        elif touch_sl:
            labels.iloc[i] = -1 # Sell (Loss)
            
        else:
            labels.iloc[i] = 0 # Hold (Timeout)
            
    return labels