# Comment Category Prediction Challenge

An end-to-end machine learning pipeline that predicts the platform-assigned category of user comments, built for a Kaggle competition on a 198,000-comment dataset.

> 🔗 Original competition: [https://www.kaggle.com/competitions/comment-category-prediction-challenge]

## 📋 Problem Statement

Each comment on an online discussion platform is automatically assigned one of **4 categories** (`label`: 0–3) based on its content and metadata. The task is to predict this label using a mix of:
- **Text data** — the raw comment content
- **Numerical signals** — upvotes, downvotes, and two hidden internal features (`if_1`, `if_2`)
- **Categorical/topic flags** — emoticon groups, and detected references to race, religion, gender, and disability

The dataset is **highly imbalanced**: Label 0 makes up ~60% of the training data, while Label 3 accounts for less than 3%.

## 🔍 Exploratory Data Analysis

- Training set: 198,000 rows × 15 columns | Test set: 102,000 rows × 14 columns
- `race`, `religion`, and `gender` were ~73.4% missing — handled with a custom binary "presence" flag instead of standard one-hot encoding
- Numerical columns (`upvote`, `downvote`, `if_1`, `if_2`) had extreme outliers, addressed using `RobustScaler`
- Word cloud analysis showed comments are highly political/opinionated in nature, reinforcing the value of a strong text-based model

## 🛠️ Feature Engineering

- Extracted `hour` and `day_of_week` from the comment timestamp
- Engineered `word_count`, `uppercase_count`, and `exclamation_count` from raw comment text
- Converted missing `race`/`religion`/`gender` values into binary presence indicators rather than dropping or imputing
- Combined two TF-IDF vectorizers: a word-level vectorizer (unigrams + bigrams, 25,000 features) and a character-level vectorizer (3–6 char n-grams, 25,000 features) to capture both semantic and stylistic text patterns

## 🤖 Models Built & Compared

| Model | Macro F1 | Accuracy | Notes |
|---|---|---|---|
| Logistic Regression | 0.81 | 0.91 | Strong baseline; best single model |
| LightGBM | 0.80 | – | Higher recall on rare Label 3, lower precision |
| LinearSVC | 0.70 | – | Underperformed relative to the other two |
| **Tuned Logistic Regression** (RandomizedSearchCV) | 0.81 | – | Confirmed baseline hyperparameters were near-optimal |
| **Voting Ensemble** (Tuned LogReg + LightGBM, soft voting) | **0.83** | **0.92** | 🏆 Final model |

The final model is a **soft-voting ensemble** of the tuned Logistic Regression and LightGBM classifiers, which outperformed every individual model by letting each cover the other's weaknesses — notably boosting both precision and recall on the rare Label 3 class.

## ⚙️ Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `LightGBM` `Matplotlib` `Seaborn` `WordCloud`

## 📁 Repository Contents

- `23f3001212-notebook-t12026.ipynb` — Full analysis, feature engineering, model building, and final submission pipeline

## 📈 Key Takeaways

- Class imbalance was handled through `class_weight` tuning rather than resampling, preserving the natural data distribution
- A custom binary-presence encoding for near-empty categorical columns avoided the noise of standard one-hot encoding on 73%-missing data
- Ensembling two structurally different models (linear + tree-based) gave a meaningful lift over any single model, especially on the hardest minority class
