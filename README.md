#  Car Price Prediction Using Machine Learning

A complete end-to-end Machine Learning pipeline to predict car prices from 26 features including engine specs, body dimensions, brand, and fuel type. Seven models are trained, evaluated, and compared — with **Random Forest achieving R² = 0.9449**.

---

## 📊 Results at a Glance

| Model | R² Score | RMSE ($) | CV R² |
|---|---|---|---|
| **Random Forest** | **0.9449** | **$1,872** | **0.9019** |
| Gradient Boosting (Tuned) | 0.9266 | $2,529 | 0.9057 |
| Ridge Regression | 0.9032 | $2,815 | 0.9195 |
| Lasso Regression | 0.8990 | $2,891 | 0.9144 |
| Linear Regression | 0.8914 | $3,075 | 0.9014 |
| Decision Tree | 0.8539 | $2,896 | 0.8451 |
| SVR | 0.6732 | $5,401 | 0.7250 |

---

##  Problem Statement

Accurately pricing a car is a complex task that depends on dozens of technical and categorical factors. This project builds a predictive model that estimates car prices from physical and mechanical attributes, helping buyers, sellers, and dealers make data-driven pricing decisions.

---

##  Dataset

- **Source:** [UCI / Kaggle Car Price Dataset](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction)
- **Records:** 205 cars
- **Features:** 26 original + 7 engineered = 33 total
- **Target:** Car Price (USD) — range: $5,118 → $45,400

### Feature Categories

| Category | Features |
|---|---|
| Identity | CarName, symboling |
| Engine | enginesize, enginetype, cylindernumber, horsepower, fuelsystem, boreratio, stroke, compressionratio |
| Body | carbody, carlength, carwidth, carheight, wheelbase, doornumber |
| Performance | citympg, highwaympg, peakrpm |
| Drivetrain | drivewheel, enginelocation |
| Weight | curbweight |
| Fuel | fueltype, aspiration |

---

## 🔧 Feature Engineering

7 new features were engineered to improve predictive power:

| Feature | Description |
|---|---|
| `brand` | Extracted brand name from CarName (with typo correction) |
| `power_to_weight` | horsepower / curbweight — performance ratio |
| `engine_to_weight` | enginesize / curbweight — engine efficiency |
| `car_volume` | length × width × height — total physical size |
| `avg_mpg` | Average of city and highway MPG |
| `bore_stroke_ratio` | boreratio / stroke — engine geometry |
| `is_luxury` | Binary flag for luxury brands (BMW, Mercedes, Porsche, etc.) |

> **Key insight:** Log-transforming the price target reduced skewness and improved model R² by ~5%.

---

## 📊 EDA Highlights

- `curbweight`, `enginesize`, and `horsepower` show the strongest correlation with price
- Luxury brands (BMW, Porsche, Mercedes) command significantly higher prices
- Hardtop and convertible body types tend to be more expensive
- Rear-wheel drive (RWD) vehicles are priced higher on average

---

##  Models Used

| Model | Key Hyperparameters |
|---|---|
| Linear Regression | Default |
| Ridge Regression | alpha=10 |
| Lasso Regression | alpha=0.001 |
| Decision Tree | random_state=42 |
| Random Forest | n_estimators=200 |
| Gradient Boosting | n_estimators=200 |
| SVR | kernel=rbf, C=100, gamma=0.1 |

Gradient Boosting was further tuned with **GridSearchCV** over 54 parameter combinations.

---

##  Project Structure

```
car-price-prediction/
│
├── notebook/
│   └── Car_Price_Prediction.ipynb    # Full ML pipeline notebook
│
├── data/
│   └── CarPrice_Assignment.csv       # Dataset
│
├── images/
│   ├── 01_price_distribution.png
│   ├── 02_price_by_brand.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_feature_vs_price.png
│   ├── 05_price_by_category.png
│   ├── 06_model_comparison.png
│   ├── 07_actual_vs_predicted.png
│   ├── 08_residuals.png
│   └── 09_feature_importance.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---
## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Programming language |
| pandas | Data manipulation |
| NumPy | Numerical computing |
| Matplotlib & Seaborn | Visualization |
| scikit-learn | ML models, preprocessing, evaluation |
| Jupyter Notebook | Interactive development |

---

## 🔑 Key Findings

1. **`curbweight` is the single strongest predictor** of car price (importance: 0.5451)
2. **Log-transforming the target** gives ~5% R² improvement over raw price
3. **Feature engineering** (power-to-weight, luxury flag) meaningfully improved model accuracy
4. **Random Forest** achieved the best test R² of 0.9449, explaining 94.5% of price variance
5. Luxury brands, rear-wheel drive, and larger engine size all strongly correlate with higher prices

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Hamza Ali Malik**
- GitHub: [@hazmaali9708](https://github.com/hazmaali9708)
- LinkedIn: [Hamza Malik](https://www.linkedin.com/in/hamza-malik-a950192a7)
- Kaggle: [hamzaali89](https://www.kaggle.com/hamzaali89)

---

>  If you found this project useful, consider giving it a star!
