# 🏈 NFLRadar: Predicting NFL Game Outcomes with Machine Learning

**Author:** Basit Umair  
**Built with:** Python, XGBoost, Pandas, Matplotlib, BeautifulSoap

---

## 📘 Overview

NFLRadar is a data-driven project that predicts NFL game results using machine learning.  
The pipeline scrapes game data from [Pro Football Reference](https://www.pro-football-reference.com/),  
calculates rolling team statistics, and trains an XGBoost regression model  
to predict future game outcomes and scores.

---

## 🧠 Pipeline Overview

1. **Data Collection (`fetch_pfr.ipynb`)**  
   - Scrapes 5 seasons of NFL data using BeautifulSoup.  
   - Cleans, renames, and exports structured datasets to `data/`.

2. **Model Training & Prediction (`prediction.ipynb`)**  
   - Builds team-level rolling stats (3-game average).  
   - Accounts for opponent stats and rest days.  
   - Trains an XGBoost MultiOutputRegressor to predict full box-score stats.  
   - Outputs predictions to `out/nfl_predictions.csv`.

3. **Visualization (`visualization.ipynb`)**  
   - Displays season-wide prediction trends with Matplotlib.  
   - Plots team performance, predicted wins, and score distributions.

---

## 🧩 Development Notes & Model Insights

Building the **NFLRadar** prediction model was a multi-stage process involving iteration, debugging, and refinement.  
Below are some of the most notable insights and lessons learned during development.

---

### ⚙️ 1️⃣ Feature Engineering
- **Opponent stats and rolling averages:**  
  Including both team and opponent 3-game rolling averages stabilized the model and prevented it from “cheating” by seeing future data.  
- **Rest days:**  
  Adding `rest_days` and `rest_days_opp` (days since last game) improved predictive realism and boosted accuracy from ~55% to **~63%**, reflecting the real-world effects of recovery and fatigue.

---

### 🧠 2️⃣ Model Training Challenges
- **NaN values and missing data:**  
  Some early-season games lacked complete stats. The model now uses *prior-only averages* (no future data leakage) and falls back to league means where necessary.  
- **Accuracy mismatch:**  
  The terminal displays ~50% accuracy because future games don’t have real results yet (`NaN` values).  
  When evaluated only on known games, the true accuracy is about **63%**.  
- **Edge cases:**  
  - Week 18 initially disappeared due to timezone and date parsing issues — fixed by explicitly setting all games to year `2025`.  
  - Rounding predicted scores slightly reduced accuracy, so rounding is deferred to visualization instead.

---

### 📊 3️⃣ Final Model Configuration
```python
XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.7,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=1
)
```
---

### 💡 4️⃣ Lessons Learned
- Preventing data leakage (using only prior stats) was crucial for trustworthy results.
- Adding more data doesn’t always help — feature relevance > feature quantity.
- Balancing interpretability and accuracy (XGBoost vs. neural nets) matters for maintainability.

---

## ⚙️ Setup

```bash
git clone https://github.com/YOUR_USERNAME/NFLRadar.git
cd NFLRadar
pip install -r requirements.txt
