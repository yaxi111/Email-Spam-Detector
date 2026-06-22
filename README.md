# Email Spam Detector

**Course:** SAIA 2163 — Natural Language Processing  
**Theme:** Theme 4 — Email/SMS Spam Detector  
**Dataset:** [SMS Spam Collection (UCI / Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

## Team Members

| Name | Student ID   | Matric Number |
|------|--------------|---------------|
| Member 1 | Cheng Yaxi   | A24AI4001 |
| Member 2 | Ding Shuna   | A24AI4008 |
| Member 3 | Yang Zhenhao | A24AI4013 |
| Member 4 | Hua Yipin    | A24AI4015 |

## Project Overview

This project builds a machine learning pipeline to automatically classify SMS messages as **Spam** or **Ham (legitimate)** using NLP techniques.

**Pipeline:**
1. Text Preprocessing (lowercase, remove special chars, stopword removal, stemming)
2. Feature Extraction — TF-IDF & Bag of Words
3. Model Training — Naive Bayes & Logistic Regression
4. Evaluation — Accuracy, Precision, Recall, F1-Score, Confusion Matrix
5. Streamlit Web App for live prediction

## Project Structure
```
Email-Spam-Detector/
├── spam.csv                  # SMS Spam Collection dataset
├── models/
│   ├── lr_model.pkl              # Trained Logistic Regression model
│   ├── nb_model.pkl              # Trained Naive Bayes model
│   └── tfidf_vectorizer.pkl      # Fitted TF-IDF vectorizer
├── notebooks/
│   └── Group_6_NLP_.ipynb        # Full NLP pipeline notebook
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
└── README.md
```

## How to Run

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Run the Streamlit app
```
streamlit run app.py
```

## Models & Results

| Model | Features | Accuracy | F1-Score |
|-------|----------|----------|----------|
| Naive Bayes | TF-IDF | ~97% | ~97% |
| Naive Bayes | Bag of Words | ~97% | ~97% |
| Logistic Regression | TF-IDF | ~98% | ~98% |
| Logistic Regression | Bag of Words | ~98% | ~97% |

Best model: **Logistic Regression + TF-IDF**

## Visualizations

1. Class Distribution (Spam vs Ham)
2. Word Cloud (Spam vs Ham)
3. Message Length Distribution
4. Confusion Matrix
5. Model Performance Comparison
6. Top Discriminative Words

## Libraries Used

- `scikit-learn` — ML models, TF-IDF, BoW, evaluation
- `nltk` — Stopwords, stemming
- `pandas` / `numpy` — Data processing
- `matplotlib` / `seaborn` — Visualizations
- `wordcloud` — Word cloud generation
- `streamlit` — Web application
- `joblib` — Save/load models
