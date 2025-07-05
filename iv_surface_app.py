# ============================ #
#        IMPORT LIBRARIES     #
# ============================ #
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go

from jax.scipy.stats import norm as jnorm
import jax.numpy as jnp
from jax import grad

# ============================ #
#   BLACK-SCHOLES & IV SOLVER #
# ============================ #
def black_scholes(S, K, T, r, sigma, q=0, otype="call"):
    d1 = (jnp.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    if otype == "call":
        return S * jnp.exp(-q * T) * jnorm.cdf(d1, 0, 1) - K * jnp.exp(-r * T) * jnorm.cdf(d2, 0, 1)
    else:
        return K * jnp.exp(-r * T) * jnorm.cdf(-d2, 0, 1) - S * jnp.exp(-q * T) * jnorm.cdf(-d1, 0, 1)

def loss_func(S, K, T, r, sigma_guess, price, q=0, otype="call"):
    theoretical_price = black_scholes(S, K, T, r, sigma_guess, q=q, otype=otype)
    return theoretical_price - price

loss_grad_func = grad(loss_func, argnums=4)

def solve_for_iv(S, K, T, r, price, sigma_guess=0.5, q=0, otype="call",
                 N_iter=20, epsilon=0.001, verbose=False):
    sigma = sigma_guess
    for i in range(N_iter):
        loss_val = loss_func(S, K, T, r, sigma, price, q=q, otype=otype)
        if abs(loss_val) < epsilon:
            return float(sigma)
        grad_val = loss_grad_func(S, K, T, r, sigma, price, q=q, otype=otype)
        if grad_val == 0:
            break
        sigma = sigma - loss_val / grad_val
    return np.nan

# ============================ #
#        STREAMLIT APP        #
# ============================ #
st.set_page_config(layout="wide")
st.title("Implied Volatility Surface Dashboard")

# === Sidebar Controls === #
with st.sidebar:
    st.header("Parameters")

    ticker_input = st.text_input("Ticker Symbol", value="SPY")

    r_input = st.slider(
        "Risk-free Rate (Annualized)",
        min_value=0.000,
        max_value=0.100,
        value=0.050,
        step=0.001,
        format="%.3f"
    )

    date_limit = st.slider("Max Expiration Dates", 1, 10, 5)

    generate = st.button("Generate IV Surface")

# === Main Logic === #
if generate:
    try:
        ticker = yf.Ticker(ticker_input.upper())
        spot = ticker.history(period='1d')['Close'][-1]
        all_dates = ticker.options
        dates = all_dates[:date_limit]

        moneyness = []
        dtes = []
        ivs = []

        for date in dates:
            try:
                call_chain = ticker.option_chain(date).calls
            except:
                continue

            expiry = datetime.strptime(date, "%Y-%m-%d")
            days_to_expiry = (expiry - datetime.today()).days
            T = days_to_expiry / 365.0
            if T <= 0:
                continue

            for _, row in call_chain.iterrows():
                strike = row.get('strike')
                market_price = row.get('lastPrice')

                if (np.isnan(market_price) or market_price <= 0 or strike == 0 or
                    abs(strike - spot) / spot > 0.2):
                    continue

                try:
                    iv = solve_for_iv(S=spot, K=strike, T=T, r=r_input, price=market_price)
                    if np.isnan(iv) or iv <= 0 or iv > 5:
                        continue
                    m = spot / strike
                    moneyness.append(m)
                    dtes.append(days_to_expiry)
                    ivs.append(iv)
                except:
                    continue

        moneyness = np.array(moneyness)
        dtes = np.array(dtes)
        ivs = np.array(ivs)

        st.markdown(f"**Data points:** {len(moneyness)} &nbsp;|&nbsp; Unique moneyness: {len(np.unique(moneyness))} &nbsp;|&nbsp; Unique expirations: {len(np.unique(dtes))}")

        if len(moneyness) < 3 or len(np.unique(moneyness)) < 3 or len(np.unique(dtes)) < 3:
            st.warning(" Not enough diverse data points to plot the surface.")
        else:
            # Interpolation for smoother surface
            from scipy.interpolate import griddata
            df = pd.DataFrame({'moneyness': moneyness, 'dtes': dtes, 'ivs': ivs})

            grid_x, grid_y = np.meshgrid(
                np.linspace(df['moneyness'].min(), df['moneyness'].max(), 50),
                np.linspace(df['dtes'].min(), df['dtes'].max(), 50)
            )

            grid_z = griddata(
                points=(df['moneyness'], df['dtes']),
                values=df['ivs'],
                xi=(grid_x, grid_y),
                method='linear'
            )

            fig = go.Figure(data=[
                go.Surface(
                    x=grid_x,
                    y=grid_y,
                    z=grid_z,
                    colorscale='Viridis',
                    colorbar=dict(title="Implied Volatility")
                )
            ])
            fig.update_layout(
                scene=dict(
                    xaxis_title="Moneyness (S/K)",
                    yaxis_title="Days to Expiry",
                    zaxis_title="Implied Volatility"
                ),
                title="Interactive Implied Volatility Surface",
                margin=dict(l=20, r=20, b=20, t=40)
            )

            st.plotly_chart(fig, use_container_width=False)

    except Exception as e:
        st.error(f" Error: {e}")
