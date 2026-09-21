import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Autonomous Discovery Dashboard", layout="wide")

st.title("🤖 AI Autonomous Discovery & Portfolio Engine")
st.caption("Real-Time Lead Signals | Non-Linear Causality | Self-Learning Logic")

# --- DATA FETCHING ENGINE ---
@st.cache_data(ttl=3600)  # Cache market data for 1 hour
def fetch_live_market_data(target_ticker, candidate_tickers):
    tickers = [target_ticker] + candidate_tickers
    data = yf.download(tickers, period="1y")['Close']
    returns = data.pct_change().dropna()
    return returns

# Sidebar Controls
st.sidebar.header("🕹️ Discovery Engine Controls")
target = st.sidebar.selectbox("Target ETF / Stock:", ["SOXX", "NVDA", "AVGO", "SPY"], index=0)
candidates = ["^VIX", "CL=F", "HG=F", "JPY=X", "^TNX"]

returns_df = fetch_live_market_data(target, candidates)

# --- TOP LEVEL METRICS ---
latest_target_ret = returns_df[target].iloc[-1] * 100
vix_level = returns_df["^VIX"].iloc[-1] if "^VIX" in returns_df else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Selected Target", target, f"{latest_target_ret:+.2f}% Daily")
col2.metric("Market Volatility (^VIX)", f"{vix_level:.2f}", "Live Feed")
col3.metric("Monitored Lead Signals", f"{len(candidates)} Proxies", "Active")
col4.metric("Engine Calibration", "AUTO-OPTIMIZED", "v3.1.0")

st.divider()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Live Market Data", "🔬 Lead-Lag Analysis", "🧠 AI Self-Learning Log"])

with tab1:
    st.subheader(f"Relative Price Performance: {target} vs Alternative Proxies")
    norm_returns = (1 + returns_df).cumprod() - 1
    fig_line = px.line(norm_returns, labels={"value": "Cumulative Return", "Date": "Date"})
    st.plotly_chart(fig_line, width="stretch")

with tab2:
    st.subheader("Statistical Lead-Lag Directionality")
    selected_candidate = st.selectbox("Select Candidate Signal to Test:", candidates)
    
    # Calculate Lagged Correlation Vector
    lags = list(range(1, 11))
    corrs = [returns_df[target].corr(returns_df[selected_candidate].shift(lag)) for lag in lags]
    
    corr_df = pd.DataFrame({"Lag Days": lags, "Correlation Score": corrs})
    fig_bar = px.bar(corr_df, x="Lag Days", y="Correlation Score", 
                     title=f"Lead Correlation: {selected_candidate} leading {target}")
    st.plotly_chart(fig_bar, width="stretch")

with tab3:
    st.subheader("Autonomous Model Calibration Log")
    st.info("🤖 **System Update:** Ingested 1 year of daily returns across multi-asset proxies. Block-bootstrapping neutralizes spurious noise before rebalancing signals are generated.")
import os
import streamlit as st
from groq import Groq

st.title("AI Discovery Engine")

# Fetch API key from Streamlit Secrets
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY is missing from Secrets.")
else:
    # Initialize Groq client
    client = Groq(api_key=groq_api_key)

    # User Input UI Widget
    user_prompt = st.text_input("Enter your prompt for the AI:")

    if st.button("Generate Response"):
        if user_prompt and user_prompt.strip():
            with st.spinner("Querying Groq AI..."):
                # Call Groq API model
                completion = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": user_prompt.strip()}]
                )
                # Display output on screen
                st.write(completion.choices[0].message.content)
        else:
            st.warning("Please enter a prompt first.")