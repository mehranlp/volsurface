# 📈 Interactive Implied Volatility Surface Dashboard

This project is a fully interactive web-based dashboard that visualizes the **implied volatility surface** of call options for any U.S. stock ticker. The dashboard is built using **Streamlit** and **Plotly**, and computes implied volatilities using a custom **Black-Scholes + JAX-based Newton-Raphson solver** — not market-provided IVs.

---

## Features

-  **User-selectable stock ticker**
-  **Custom implied volatility solver** using JAX automatic differentiation
-  Computes implied volatility by solving the Black-Scholes formula from option prices
-  Adjustable number of expiration dates
-  Filters near-the-money call options for cleaner, relevant surfaces
-  **Interactive 3D Plotly surface**: zoom, rotate, and hover
-  Optimized performance with caching and smart filtering

---

## Implied Volatility Calculation

Implied volatility is computed using a **custom Newton-Raphson root-finding algorithm** applied to the Black-Scholes pricing formula:

```python
sigma = sigma - f(sigma) / f'(sigma)
