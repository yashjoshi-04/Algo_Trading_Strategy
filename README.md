# Algo Trading Strategy Leveraging ARIMA-X

---

## 📖 **Introduction**

I aimed to build an algorithmic trading strategy that could **outperform SPY returns and CAGR** by timing buy and sell signals exclusively for SPY. 

This journey was not without failures. Before finding a successful strategy, I developed six different models that massively underperformed. My intent here is not to glorify quantitative trading but to provide a transparent view into the iterative process of creating a viable strategy.

> **Note:** More often than not, trading strategies fail before finding one that works.

Below is a snapshot comparing two failed strategies:
- **Red Line**: Mean Reversion using Gold Futures
- **Blue Line**: Simple MACD Strategy
- **Orange Line**: SPY Cumulative Returns (baseline)

<img width="827" alt="Screenshot 2024-11-20 at 1 12 14 PM" src="https://github.com/user-attachments/assets/45a3ab74-6f0d-48fc-9d4e-57243098a10e">

---

## 🧐 **Rationale**

The objective was to leverage the **inverse relationship between commodities and equities** to generate buy/sell signals for SPY using adjusted closing prices of key indicators. 

### **Key Insights:**
1. **Gold and Silver Futures**: Not volatile enough to provide meaningful signals for SPY.
2. **Energy Futures (yfinance: `ES=F`)**: Exhibited both high volatility and a strong inverse correlation with SPY, making it a promising indicator.
3. **VIX Index (yfinance: `^VIX`)**: Captures downward volatility more effectively than SPY, allowing prediction of market downturns before significant impact.

---

## 🛠 **Methodology**

### **Model Overview:**
- Developed an **ARIMAX model** with:
  - `^VIX` and `ES=F` as independent features.
  - Lagged values of `^VIX` to enhance trend capture.

### **Process:**
1. Train the ARIMAX model to capture the trend of SPY returns.
2. Overlay the **ARIMAX Predicted Trend** onto SPY actual cumulative returns.
3. Use the modeled relationship to generate **buy/sell signals**.

### **Visualization:**
Below is a comparison between the **ARIMAX Predicted Mean (Trend)** and **SPY Cumulative Returns**:
- **Red Line**: ARIMAX Predicted Trend
- **Blue Dashed Line**: SPY Cumulative Returns

<img width="1004" alt="Screenshot 2024-11-23 at 10 53 12 AM" src="https://github.com/user-attachments/assets/835a486e-7c0a-47a2-921b-c615f5c7e6ba">

---

## 📊 **Performance Metrics**

After simulating the strategy with an initial capital of **$100,000**, the following results were observed:

### **Key Metrics**:
<img width="341" alt="Screenshot 2024-11-23 at 10 59 45 AM" src="https://github.com/user-attachments/assets/29bcfc97-3b93-405b-8dfc-17f3f63c214e">

### **Portfolio Value Comparison:**
Below is a comparison between the **SPY Portfolio Value** and the **Strategy Portfolio Value**:
<img width="633" alt="Screenshot 2024-11-23 at 10 59 59 AM" src="https://github.com/user-attachments/assets/9770f602-304d-4a6a-a303-81605aa3bb32">

---

## 💡 **Conclusion**

This project highlights:
- The iterative nature of developing trading strategies.
- The importance of combining multiple indicators (`ES=F`, `^VIX`) for robust predictions.
- The use of **ARIMAX models** to generate actionable insights for SPY trading.

While the strategy succeeded in outperforming SPY, its real-world application requires continuous adaptation to dynamic market conditions.

For further inquiries or collaboration, feel free to reach out.

---
