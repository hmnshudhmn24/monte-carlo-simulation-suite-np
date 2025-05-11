# 🎲 Monte Carlo Simulation Suite (NumPy + Streamlit)

This project provides an interactive suite of Monte Carlo simulations using NumPy, including Pi estimation and stock price modeling. It supports parallel execution with Joblib and dynamic visualizations via Streamlit and Matplotlib.

## 🔬 Simulations Included

- **Estimate π** using the Monte Carlo circle method
- **Stock Price Simulation** using geometric Brownian motion

## ⚙️ Technologies

- NumPy for vectorized computation
- Joblib for parallel processing
- Matplotlib for plotting
- Streamlit for interactive UI

## 🚀 How to Run

1️⃣ Install required libraries:

```bash
pip install numpy matplotlib streamlit joblib
```

2️⃣ Launch the Streamlit app:

```bash
streamlit run src/main.py
```

3️⃣ Use the UI to select a simulation and adjust parameters.

## 📂 Project Structure

```
monte_carlo_simulation_suite_np/
├── src/
│   └── main.py        # Simulation app with Streamlit
├── README.md          # Documentation
```

## 💡 Innovative Twist

- Dynamically adjust simulation parameters
- Visual output of multiple stock price paths
- Run simulations in parallel with Joblib for performance boost

---

🔁 Simulate. Visualize. Optimize.