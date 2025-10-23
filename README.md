# ⚡ DaysAhead: Dutch Electricity Price Forecasting

**DaysAhead** is a machine learning project designed to predict **Dutch electricity prices** up to **48 hours into the future**.  
The code leverages historical energy market data and ML modeling techniques to forecast short-term price fluctuations. The project was started to help with determining strategies for the https://github.com/corneel27/day-ahead project.

---

## 📊 Project Overview

Electricity prices in the Netherlands (and broader European market) are highly volatile due to factors such as weather, renewable energy generation, and demand fluctuations.  
**DaysAhead** aims to capture these dynamics using predictive modeling techniques, providing accurate forecasts for the next 48 hours.

Key objectives:
- Collect and preprocess historical Dutch electricity market data  
- Explore price trends and correlations with external factors  
- Train a regression model to predict future prices  
- Evaluate model accuracy using relevant metrics (e.g., RMSE)  
- Visualize predictions vs. actual prices

---

## 🧠 Features

- ⏱ Predicts Dutch electricity prices **up to 48 hours ahead**  
- 📈 Uses **time series forecasting** and/or **machine learning** models  
- 🔍 Includes **data cleaning, feature engineering, and model evaluation**  
- 💡 Provides **visual insights** into price trends and forecast performance  
- 🧾 Fully contained in a Jupyter notebook for easy reproducibility

---

## 🧰 Technologies Used

| Category | Tools / Libraries |
|-----------|------------------|
| Language | Python |
| Environment | Jupyter Notebook |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn, plotly |
| Modeling | scikit-learn, XGBoost, statsmodels (depending on implementation) |
| Evaluation | RMSE, MAE, R² |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/jsimonetti/daysahead.git
cd daysahead
```

### 2. Install Dependencies
It’s recommended to use a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Notebook
Open Jupyter and execute:
```bash
jupyter notebook daysahead.ipynb
```
Or when running on a remote machine:
```
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

---

## 📈 Example Output

The notebook outputs:
- Forecast plots comparing predicted vs. actual electricity prices  
- Evaluation metrics for forecast accuracy  
- (Optional) Interactive visualizations for exploring trends

---

## 🧩 Project Structure

```
daysahead/
│
├── daysahead.ipynb           # Main notebook with data prep, model, and results
├── 2day_predict.py           # Example usage
├── features.py               # Adding features
├── daysahead_xgb_model.json  # trained model, ready for usage
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

---

## 🤝 Contributing

Contributions are welcome!  
If you’d like to improve the model or add new features:
1. Fork the repository  
2. Create a new branch  
3. Submit a pull request  

---

## 🪪 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

Jeroen Simonetti  
🔗 [GitHub](https://github.com/jsimonetti)

---

> *Forecast day after tomorrow’s power prices today – with DaysAhead ⚡*
