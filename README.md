\# 🏦 Credit Scoring Engine



> A production-grade, bank-standard credit scoring system that takes raw loan applicant data and returns a \*\*credit score (300–900)\*\*, risk category, default probability, scorecard breakdown, and SHAP explanations — the exact methodology used by RBI-regulated banks and CIBIL.



\---



\## 🏗️ System Architecture



!\[System Architecture](docs/architecture.png)



\---



\## ✨ What Makes This Different



Most ML projects stop at a Jupyter notebook with accuracy score. This one goes further:



\- ✅ \*\*Weight of Evidence (WoE)\*\* encoding — the industry standard for credit features

\- ✅ \*\*Information Value (IV)\*\* feature selection — dropped 61 useless features, kept 30 strong ones

\- ✅ \*\*Logistic Regression Scorecard\*\* scaled to 300–900 — explainable, auditable, regulation-friendly

\- ✅ \*\*SHAP explanations\*\* — every prediction explained per feature

\- ✅ \*\*Bank-standard metrics\*\* — Gini coefficient, KS statistic, PSI — not just accuracy

\- ✅ \*\*FastAPI REST API\*\* — callable by anyone, anywhere

\- ✅ \*\*Docker\*\* — fully containerized and ready to deploy



\---



\## 📊 Model Performance



| Metric | Train | Test |

|--------|-------|------|

| AUC | 0.7409 | 0.7446 |

| Gini Coefficient | 0.4818 | 0.4891 |

| KS Statistic | 36.18 | 36.72 |

| PSI | — | 0.0001 ✅ Stable |



> Gini above 0.4 is the industry acceptance threshold for bank scorecards. KS above 30 is considered good separation.



\---



\## 🎯 Risk Categories



| Score Range | Category | Default Rate |

|-------------|----------|--------------|

| 800 – 900 | 🟢 Excellent | \~0.0% |

| 720 – 799 | 🔵 Good | \~1.4% |

| 650 – 719 | 🟡 Fair | \~2.3% |

| 580 – 649 | 🟠 Poor | \~5.0% |

| 300 – 579 | 🔴 Very Poor | \~15.8% |



\---



\## 📈 EDA Insights



\### Target Distribution

!\[Target Distribution](docs/target\_distribution.png)



> 91.9% repaid vs 8.1% defaulted — 11:1 class imbalance handled using class weights in Logistic Regression.



\### Default Rate by Key Features

!\[Default Rate Categorical](docs/default\_rate\_categorical.png)



\### Correlation Heatmap

!\[Correlation Heatmap](docs/correlation\_heatmap.png)



\---



\## 🔬 WoE \& IV Analysis



\### Top Features by Information Value

!\[WoE Top 6 Features](docs/woe\_top6\_features.png)



> EXT\_SOURCE\_2 and EXT\_SOURCE\_3 (external credit bureau scores) are the strongest predictors with IV above 0.30 — classified as Strong predictors.



\---



\## 📉 Model Evaluation



\### ROC Curve — All Models Compared

!\[ROC Curve Comparison](docs/roc\_curve\_comparison.png)



> Logistic Regression chosen over XGBoost and LightGBM despite marginally lower AUC because it is fully explainable, auditable by regulators, and required for scorecard scaling.



\---



\## 🎰 Scorecard Distribution



!\[Scorecard Distribution](docs/scorecard\_distribution.png)



> Default rates increase monotonically across risk categories — confirming the scorecard discriminates correctly.



\---



\## 🚀 API Usage



\### Run Locally



```bash

\# Clone the repo

git clone https://github.com/saicharan8855/credit-scoring-engine.git

cd credit-scoring-engine



\# Install dependencies

pip install -r requirements.txt



\# Start the API

uvicorn api.main:app --reload --port 8000

```



\### API Endpoints



| Method | Endpoint | Description |

|--------|----------|-------------|

| GET | `/health` | Check if API is running |

| POST | `/predict` | Get full credit assessment |

| POST | `/predict/explain` | Get assessment with plain English explanation |

| GET | `/docs` | Swagger UI documentation |



\### Example Request



```bash

curl -X POST http://127.0.0.1:8000/predict \\

&#x20; -H "Content-Type: application/json" \\

&#x20; -d '{

&#x20;   "AMT\_CREDIT": 500000,

&#x20;   "AMT\_ANNUITY": 25000,

&#x20;   "AMT\_INCOME\_TOTAL": 180000,

&#x20;   "AMT\_GOODS\_PRICE": 450000,

&#x20;   "CODE\_GENDER": "M",

&#x20;   "DAYS\_BIRTH": -12000,

&#x20;   "DAYS\_EMPLOYED": -3000,

&#x20;   "EXT\_SOURCE\_2": 0.6,

&#x20;   "EXT\_SOURCE\_3": 0.5,

&#x20;   "NAME\_EDUCATION\_TYPE": "Higher education",

&#x20;   "NAME\_INCOME\_TYPE": "Working",

&#x20;   "NAME\_FAMILY\_STATUS": "Married",

&#x20;   "NAME\_HOUSING\_TYPE": "House / apartment",

&#x20;   "NAME\_CONTRACT\_TYPE": "Cash loans",

&#x20;   "FLAG\_OWN\_CAR": "Y",

&#x20;   "FLAG\_OWN\_REALTY": "Y",

&#x20;   "CNT\_CHILDREN": 0,

&#x20;   "CNT\_FAM\_MEMBERS": 2,

&#x20;   "DAYS\_REGISTRATION": -5000,

&#x20;   "DAYS\_ID\_PUBLISH": -2000,

&#x20;   "DAYS\_LAST\_PHONE\_CHANGE": -500,

&#x20;   "REGION\_RATING\_CLIENT": 2,

&#x20;   "REGION\_RATING\_CLIENT\_W\_CITY": 2,

&#x20;   "REGION\_POPULATION\_RELATIVE": 0.035,

&#x20;   "OCCUPATION\_TYPE": "Laborers",

&#x20;   "ORGANIZATION\_TYPE": "Business Entity Type 3",

&#x20;   "TOTALAREA\_MODE": 0.05,

&#x20;   "FLOORSMAX\_AVG": 0.1,

&#x20;   "FLOORSMAX\_MODE": 0.1,

&#x20;   "FLOORSMAX\_MEDI": 0.1,

&#x20;   "YEARS\_BEGINEXPLUATATION\_AVG": 0.9,

&#x20;   "YEARS\_BEGINEXPLUATATION\_MODE": 0.9,

&#x20;   "YEARS\_BEGINEXPLUATATION\_MEDI": 0.9,

&#x20;   "OWN\_CAR\_AGE": 5,

&#x20;   "EXT\_SOURCE\_1": 0.5

&#x20; }'

```



\### Example Response



```json

{

&#x20; "credit\_score": 596,

&#x20; "risk\_category": "Poor",

&#x20; "default\_probability": 0.4409,

&#x20; "scorecard\_breakdown": \[

&#x20;   {"feature": "Base Score (Intercept)", "points": 603.95},

&#x20;   {"feature": "NAME\_EDUCATION\_TYPE", "points": 19.04},

&#x20;   {"feature": "CREDIT\_TO\_ANNUITY\_RATIO", "points": 12.09}

&#x20; ],

&#x20; "shap\_explanation": \[

&#x20;   {"feature": "NAME\_EDUCATION\_TYPE", "shap\_value": -0.2728, "impact": "decreases default risk"},

&#x20;   {"feature": "AMT\_GOODS\_PRICE", "shap\_value": 0.2681, "impact": "increases default risk"}

&#x20; ],

&#x20; "top\_risk\_factors": \["AMT\_GOODS\_PRICE", "CODE\_GENDER", "EXT\_SOURCE\_3"],

&#x20; "top\_strengths": \["NAME\_EDUCATION\_TYPE", "CREDIT\_TO\_ANNUITY\_RATIO", "AMT\_CREDIT"]

}

```



\---



\## 🐳 Docker



```bash

\# Build the image

docker build -t credit-scoring-engine .



\# Run the container

docker run -p 8000:8000 credit-scoring-engine

```



\---



\## 📁 Project Structure

credit-scoring-engine/

├── data/

│   ├── raw/                    ← original dataset (not tracked)

│   ├── processed/              ← WoE transformed train/test sets

│   └── external/

├── notebooks/

│   ├── 01\_eda.ipynb

│   ├── 02\_woe\_iv\_analysis.ipynb

│   ├── 03\_model\_training.ipynb

│   ├── 04\_scorecard.ipynb

│   └── 05\_shap\_explainability.ipynb

├── src/

│   ├── data/

│   │   ├── ingestion.py        ← load and split data

│   │   └── preprocessing.py   ← full cleaning pipeline

│   ├── features/

│   │   ├── woe\_encoder.py      ← WoE transformer

│   │   └── iv\_selector.py     ← IV feature selector

│   ├── models/

│   │   ├── scorecard.py        ← 300-900 scaling logic

│   │   ├── train.py            ← training pipeline

│   │   └── evaluate.py        ← Gini, KS, PSI metrics

│   └── explainability/

│       └── shap\_explainer.py  ← SHAP wrapper

├── api/

│   ├── main.py                 ← FastAPI app

│   ├── schemas.py              ← request/response models

│   ├── predictor.py            ← inference pipeline

│   └── explainer.py           ← plain English explanations

├── models/                     ← saved pkl files (not tracked)

├── docs/                       ← plots and architecture diagram

├── Dockerfile

├── requirements.txt

└── README.md

\---



\## 🛠️ Tech Stack



| Category | Tools |

|----------|-------|

| Data Processing | pandas, numpy |

| Machine Learning | scikit-learn, XGBoost, LightGBM |

| WoE / Scorecard | Custom implementation using pandas |

| Explainability | SHAP |

| Experiment Tracking | MLflow |

| API | FastAPI, Uvicorn, Pydantic |

| Containerization | Docker |

| Dataset | Home Credit Default Risk (Kaggle) |



\---



\## 📦 Dataset



\*\*Home Credit Default Risk\*\* — Kaggle Competition



\- 307,511 loan applications

\- 122 raw features

\- 8.07% default rate

\- Download: https://www.kaggle.com/competitions/home-credit-default-risk/data



\---



\## 📜 License



MIT License — see \[LICENSE](LICENSE) for details.

