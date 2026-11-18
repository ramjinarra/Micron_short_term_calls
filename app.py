import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm

st.set_page_config(page_title="⚡ MU & MUU Advanced GEX Terminal", layout="wide")
st.title("⚡ MU & MUU Dynamic Options & GEX Scanner")

# 1. Wrap sidebar controls in a form to prevent API spam on slider drag
with st.sidebar.form("scanner_settings"):
    target_move_pct = st.slider("Target Move (%)", min_value=1.0, max_value=25.0, value=5.0, step=0.5) / 100.0
    risk_free_rate = st.number_input("Risk-Free Rate (e.g., 0.045 for 4.5%)", value=0.045, format="%.3f")
    dte = st.number_input("Days to Expiration (DTE)", value=1.0, min_value=0.1, step=1.0)
    submitted = st.form_submit_button("Scan Options")

tickers = ["MU", "MUU"]

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0 or pd.isna(sigma):
        return 0.0, 0.0, 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        return delta, gamma, theta
    return 0.0, 0.0, 0.0

# 2. Cache the Yahoo Finance request for 5 minutes (300 seconds)
@st.cache_data(ttl=300)
def fetch_yf_data(ticker_symbol):
    tkr = yf.Ticker(ticker_symbol)
    hist = tkr.history(period="1d")
    
    if hist.empty:
        return None, None
    
    spot_price = hist['Close'].iloc[-1]
    exps = tkr.options
    
    if not exps:
        return spot_price, None
        
    opt_chain = tkr.option_chain(exps[0])
    return spot_price, opt_chain.calls

def get_advanced_otm_setups(ticker_symbol, move_pct, r, days_to_expiry):
    # Fetch from cache instead of hitting the API directly
    spot_price, calls = fetch_yf_data(ticker_symbol)
    
    if spot_price is None or calls is None or calls.empty:
        return spot_price, None
        
    target_price = spot_price * (1 + move_pct)
    
    filtered_calls = calls[
        (calls['strike'] > spot_price) & 
        (calls['strike'] <= target_price * 1.02) 
    ].copy()
    
    if filtered_calls.empty:
        return spot_price, None
        
    filtered_calls = filtered_calls.sort_values(by='strike')
    
    T = days_to_expiry / 365.0
    deltas, thetas, gex, decay_ratios = [], [], [], []
    
    for _, row in filtered_calls.iterrows():
        K = row['strike']
        sigma = row['impliedVolatility']
        oi = row['openInterest']
        
        delta, gamma, theta = calculate_greeks(spot_price, K, T, r, sigma)
        gex_val = gamma * oi * 100 * spot_price
        decay_ratio = abs(theta) / delta if delta > 0 else 0
        
        deltas.append(delta)
        thetas.append(theta)
        gex.append(gex_val)
        decay_ratios.append(decay_ratio)
        
    filtered_calls['Delta'] = deltas
    filtered_calls['Theta (Daily)'] = thetas
    filtered_calls['GEX ($)'] = gex
    filtered_calls['Theta/Delta'] = decay_ratios
    
    display_df = filtered_calls[[
        'strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility', 
        'volume', 'openInterest', 'Delta', 'Theta (Daily)', 'Theta/Delta', 'GEX ($)'
    ]]
    
    return spot_price, display_df

# Only execute the heavy processing if the user clicked the form button
if submitted:
    for ticker in tickers:
        adjusted_move = target_move_pct * 2 if ticker == "MUU" else target_move_pct
        st.subheader(f"{ticker} Advanced Greek Scanner (Targeting +{adjusted_move*100:.1f}% Move)")
        
        spot, df_calls = get_advanced_otm_setups(ticker, adjusted_move, risk_free_rate, dte)
        
        if spot is None:
            st.error(f"Failed to fetch market data for {ticker}. Yahoo Finance may be temporarily blocking you.")
        elif df_calls is not None and not df_calls.empty:
            st.write(f"**Current Spot Price:** ${spot:.2f} | **Target Price:** ${spot * (1 + adjusted_move):.2f}")
            
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
else:
    st.info("Adjust your parameters in the sidebar and click **Scan Options** to pull live market data.")
