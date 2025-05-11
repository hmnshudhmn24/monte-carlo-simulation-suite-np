import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import streamlit as st

def monte_carlo_pi(num_samples):
    x = np.random.rand(num_samples)
    y = np.random.rand(num_samples)
    inside_circle = x**2 + y**2 <= 1
    return np.sum(inside_circle)

def monte_carlo_sim_pi(total_samples, num_jobs=4):
    samples_per_job = total_samples // num_jobs
    results = Parallel(n_jobs=num_jobs)(delayed(monte_carlo_pi)(samples_per_job) for _ in range(num_jobs))
    pi_estimate = 4 * sum(results) / total_samples
    return pi_estimate

def simulate_stock_price(S0, mu, sigma, T, dt, N):
    steps = int(T / dt)
    prices = np.zeros((N, steps))
    prices[:, 0] = S0
    for t in range(1, steps):
        z = np.random.standard_normal(N)
        prices[:, t] = prices[:, t-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    return prices

def main():
    st.title("🎲 Monte Carlo Simulation Suite")

    sim_type = st.selectbox("Select Simulation Type", ["Estimate Pi", "Stock Price Simulation"])

    if sim_type == "Estimate Pi":
        total_samples = st.slider("Total Samples", 1000, 10000000, 1000000, step=100000)
        num_jobs = st.slider("Parallel Jobs", 1, 8, 4)
        if st.button("Run Simulation"):
            pi_estimate = monte_carlo_sim_pi(total_samples, num_jobs)
            st.write(f"Estimated π value: {pi_estimate}")

    elif sim_type == "Stock Price Simulation":
        S0 = st.number_input("Initial Stock Price (S₀)", value=100.0)
        mu = st.number_input("Expected Return (μ)", value=0.05)
        sigma = st.number_input("Volatility (σ)", value=0.2)
        T = st.number_input("Time Horizon (T)", value=1.0)
        dt = st.number_input("Time Step (dt)", value=0.01)
        N = st.slider("Number of Simulations (N)", 10, 5000, 1000)

        if st.button("Run Simulation"):
            prices = simulate_stock_price(S0, mu, sigma, T, dt, N)
            st.write("📈 Simulated Stock Price Paths")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(prices.T, color="skyblue", alpha=0.3)
            ax.set_title("Monte Carlo Simulated Stock Prices")
            st.pyplot(fig)

if __name__ == "__main__":
    main()