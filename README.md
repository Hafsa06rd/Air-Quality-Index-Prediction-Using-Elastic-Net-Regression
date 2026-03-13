#  Air Quality Index Prediction Using Elastic Net Regression

> Predicting India's Air Quality Index (AQI) from pollutant measurements using a tuned Elastic Net regression model, with full sub-index computation following the CPCB standard formula.

---

##  Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Key Takeaways](#-key-takeaways)
- [Potential Extensions](#-potential-extensions)
- [Tech Stack](#-tech-stack)
- [License](#-license)

---

##  Project Overview

Air pollution is one of the most pressing public health challenges in India. This project builds a machine learning pipeline to **predict the Air Quality Index (AQI)** from raw atmospheric pollutant readings, using a dataset of over **435,000 monitoring records** collected across Indian states between 1990 and 2015.

The project covers the complete data science workflow: exploratory analysis, feature engineering (AQI sub-index computation), model training with hyperparameter tuning, and evaluation, all grounded in the official **Central Pollution Control Board (CPCB)** methodology.

---

##  Dataset

| Property | Detail |
|----------|--------|
| **Source** | [India Air Quality Data — Kaggle](https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data) |
| **Records** | 435,742 |
| **Time span** | 1990 – 2015 |
| **Coverage** | Multiple states, cities, and station types across India |

**Key columns used:**

| Column | Description |
|--------|-------------|
| `so2` | Sulphur Dioxide concentration (µg/m³) |
| `no2` | Nitrogen Dioxide concentration (µg/m³) |
| `rspm` | Respirable Suspended Particulate Matter (µg/m³) |
| `spm` | Suspended Particulate Matter (µg/m³) |
| `state` | Indian state of the monitoring station |
| `type` | Station type (Industrial, Residential, etc.) |

---

##  Methodology

### 1. Data Cleaning & Feature Selection
- Selected 8 relevant columns from the 13-column raw dataset
- Filled categorical nulls (`location`, `type`) with the column mode
- Filled numeric pollutant nulls with `0`, consistent with below-detection-threshold readings

### 2. AQI Sub-index Computation (CPCB Formula)
AQI is calculated using the **dominant pollutant method**: each raw concentration is mapped to a 0–500 sub-index via piecewise linear interpolation, and the final AQI is the maximum across all sub-indices.

| Sub-index | Pollutant | Function |
|-----------|-----------|----------|
| SOi | SO₂ | `cal_SOi()` |
| Noi | NO₂ | `cal_Noi()` |
| Rpi | RSPM | `cal_RSPMI()` |
| SPMi | SPM | `cal_SPMi()` |

```
AQI = max(SOi, Noi, Rpi, SPMi)
```

**AQI Categories:**

| Range | Category |
|-------|----------|
| 0 – 50 | 🟢 Good |
| 51 – 100 | 🟡 Moderate |
| 101 – 200 | 🟠 Poor |
| 201 – 300 | 🔴 Unhealthy |
| 301 – 400 | 🟣 Very Unhealthy |
| 401+ | ⚫ Hazardous |

### 3. Model : Elastic Net Regression
**Elastic Net** combines L1 (Lasso) and L2 (Ridge) regularisation, making it robust to multicollinearity between the correlated pollutant sub-indices.

Hyperparameters were tuned using **10-fold cross-validated GridSearchCV** over:
- `alpha` ∈ log-space(−5, 2, 8 values)
- `l1_ratio` ∈ {0.2, 0.4, 0.6, 0.8}

**Best configuration:** `alpha = 0.01`, `l1_ratio = 0.2`

---

##  Results

| Metric | Score |
|--------|-------|
| R² | ≈ 1.000 |
| MAE | — |
| RMSE | — |

> The near-perfect R² reflects the fact that AQI is a deterministic function of the four sub-indices used as features, confirming correct implementation of the CPCB formula. The model serves as a solid baseline for scenarios where raw concentrations are available but the sub-index computation is abstracted away.

---

##  Repository Structure

```
 air-quality-index-elastic-net/
│
├──  Air_Quality_Index_Using_Elastic_Net_Regression.ipynb   # Main notebook
├── README.md                                               # This file
└──  data/
    └── data.csv                                               # Raw dataset (download from Kaggle)
```

---

##  Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Notebook

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data) and place `data.csv` in the project root.
2. Open the notebook:
   ```bash
   jupyter notebook Air_Quality_Index_Using_Elastic_Net_Regression.ipynb
   ```
3. Run all cells in order.

---

##  Key Takeaways

- **RSPM and SPM** are historically the dominant pollutants driving poor AQI in Indian cities, particularly in industrial and urban zones.
- Elastic Net's combined penalty handled correlated features (sub-indices are all derived from the same underlying concentrations) without overfitting.
- The piecewise CPCB formula creates a non-linear mapping from raw concentrations to AQI, Elastic Net learns this mapping effectively when sub-indices are the direct input features.

---

##  Potential Extensions

- Adding temporal features (`year`, `month`, `season`) to model seasonal pollution patterns
- Training region-specific or station-type-specific models
- Comparing with tree-based models (XGBoost, Random Forest) for non-linear relationships
- Building an interactive AQI predictor using Streamlit

---

##  Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-F7931E?logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-11557c)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-4C72B0)

---

##  License

> This project is released for educational and research purposes.

---

## Note from Me :)

> This project was developed as part of an applied portfolio effort in machine learning and environmental data science, combining rigorous pollutant sub-index engineering with systematic hyperparameter tuning on a large-scale real-world dataset.
> Contributions, suggestions, and feedback are welcome, feel free to explore !
