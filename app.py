import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm

st.set_page_config(page_title="⚡ MU & MUU Advanced GEX Terminal", layout="wide")
st.title("⚡ MU & MUU Dynamic Options & GEX Scanner")

# Interactive sidebar controls
target_move_pct = st.sidebar.slider("Target Move (%)", min_value=1.0, max_value=25.0, value=5.0, step=0.5) / 100.0
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (e.g., 0.045 for 4.5%)", value=0.045, format="%.3f")
dte = st.sidebar.number_input("Days to Expiration (DTE)", value=1.0, min_value=0.1, step=1.0)

tickers = ["MU", "MUU"]

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes Greek Calculator"""
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0, 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        # Daily Theta
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        return delta, gamma, theta
    return 0.0, 0.0, 0.0

def get_advanced_otm_setups(ticker_symbol, move_pct, r, days_to_expiry):
    tkr = yf.Ticker(ticker_symbol)
    
    hist = tkr.history(period="1d")
    if hist.empty:
        return None, None
    
    spot_price = hist['Close'].iloc[-1]
    target_price = spot_price * (1 + move_pct)
    
    exps = tkr.options
    if not exps:
        return None, None
    
    # Grabbing the nearest expiration chain
    opt_chain = tkr.option_chain(exps[0])
    calls = opt_chain.calls
    
    filtered_calls = calls[
        (calls['strike'] > spot_price) & 
        (calls['strike'] <= target_price * 1.02) 
    ].copy()
    
    if filtered_calls.empty:
        return None, None
        
    filtered_calls = filtered_calls.sort_values(by='strike')
    
    # Calculate Greeks for each row
    T = days_to_expiry / 365.0
    deltas, thetas, gex, decay_ratios = [], [], [], []
    
    for _, row in filtered_calls.iterrows():
        K = row['strike']
        sigma = row['impliedVolatility']
        oi = row['openInterest']
        
        delta, gamma, theta = calculate_greeks(spot_price, K, T, r, sigma)
        
        # GEX = Gamma * Open Interest * 100 shares * Spot Price
        gex_val = gamma * oi * 100 * spot_price
        
        # How much daily Theta you pay per 1 unit of Delta
        decay_ratio = abs(theta) / delta if delta > 0 else 0
        
        deltas.append(delta)
        thetas.append(theta)
        gex.append(gex_val)
        decay_ratios.append(decay_ratio)
        
    filtered_calls['Delta'] = deltas
    filtered_calls['Theta (Daily)'] = thetas
    filtered_calls['GEX ($)'] = gex
    filtered_calls['Theta/Delta'] = decay_ratios
    
    # Reorder and isolate columns for the final terminal view
    display_df = filtered_calls[[
        'strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility', 
        'volume', 'openInterest', 'Delta', 'Theta (Daily)', 'Theta/Delta', 'GEX ($)'
    ]]
    
    return spot_price, display_df

for ticker in tickers:
    adjusted_move = target_move_pct * 2 if ticker == "MUU" else target_move_pct
    
    st.subheader(f"{ticker} Advanced Greek Scanner (Targeting +{adjusted_move*100:.1f}% Move)")
    
    spot, df_calls = get_advanced_otm_setups(ticker, adjusted_move, risk_free_rate, dte)
    
    if df_calls is not None and not df_calls.empty:
        st.write(f"**Current Spot Price:** ${spot:.2f} | **Target Price:** ${spot * (1 + adjusted_move):.2f}")
        
        # Highlight GEX and Decay logic cleanly in Streamlit
        styled_df = df_calls.style.format({
            'strike': '{:.2f}', 'lastPrice': '{:.2f}', 'bid': '{:.2f}', 'ask': '{:.2f}', 
            'impliedVolatility': '{:.2%}', 'Delta': '{:.3f}', 'Theta (Daily)': '{:.3f}', 
            'Theta/Delta': '{:.3f}', 'GEX ($)': '${:,.0f}'
        }).background_gradient(subset=['GEX ($)'], cmap='Greens') \
          .background_gradient(subset=['Theta/Delta'], cmap='Reds')
        
        st.dataframe(styled_df, use_container_width=True)
        
        if len(df_calls) >= 2:
            buy_strike = df_calls.iloc[-2]['strike']
            sell_strike = df_calls.iloc[-1]['strike']
            buy_ask = df_calls.iloc[-2]['ask']
            sell_bid = df_calls.iloc[-1]['bid']
            max_cost = buy_ask - sell_bid
            
            st.info(
                f"**Nearest Call Debit Spread:** Buy ${buy_strike}C & Sell ${sell_strike}C "
                f"| Est. Max Entry: ${max_cost:.2f}"
            )
    else:
        st.warning(f"No suitable near-term options found for {ticker} matching criteria.")
