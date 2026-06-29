"""
app.py — SMS Spam Detector (Streamlit Web Application)
Course : SAIA 2163 — Natural Language Processing
Theme  : Theme 4 — Email/SMS Spam Detector

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data (runs only once)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Page configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models
def load_models():
    """Load the saved BOW vectorizer and both classifiers."""
    bow = joblib.load('models/bow_model.pkl')
    lr  = joblib.load('models/lr_model.pkl')
    nb  = joblib.load('models/nb_model.pkl')
    return bow, lr, nb

# Load dataset
def load_data():
    """Load and prepare the SMS Spam dataset."""
    df = pd.read_csv('data/spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
    df['text_length'] = df['text'].apply(len)
    df['word_count']  = df['text'].apply(lambda x: len(x.split()))
    return df

# Text preprocessing (same function as in the notebook)
stemmer    = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    """Clean and normalize input text before prediction."""
    text   = text.lower()
    text   = re.sub(r'http\S+|www\S+', '', text)          # Remove URLs
    text   = re.sub(r'[^a-z\s]', '', text)                # Keep letters only
    tokens = text.split()
    tokens = [
        stemmer.stem(w)
        for w in tokens
        if w not in stop_words and len(w) > 1
    ]
    return ' '.join(tokens)

# Sidebar navigation
st.sidebar.title("SMS Spam Detector")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Spam Detector",
     "Visualizations"]
)

# ──────────────
# PAGE 1 — HOME
# ──────────────
if page == "Home":
    st.title("SMS Spam Detector")
    st.markdown("### SAIA 2163 — NLP Final Project | Theme 4")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        #### How to use
        1. Go to **Spam Detector** in the sidebar
        2. Type or paste any SMS / email message
        3. Click **Analyze** to get an instant prediction
        4. See the confidence score and the words that influenced the result

        #### Technical approach
        - **Dataset:** SMS Spam Collection — 5,572 messages (UCI / Kaggle)
        - **Preprocessing:** Lowercase → Remove special chars → Stopword removal → Stemming
        - **Feature Extraction:** TF-IDF & Bag of Words
        - **Models:** Naive Bayes & Logistic Regression
        """)

    with col2:
        st.markdown("#### Team Members")
        st.info("""
        **Group 6 — NLP**

        Cheng Yaxi - A24AI4001\n
        Ding Shuna - A24AI4008\n
        Yang ZhenHao - A24AI4013\n
        Hua Yipin - A24AI4015
        """)

    st.markdown("---")
    st.markdown("#### Dataset at a Glance")
    try:
        df = load_data()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Messages", f"{len(df):,}")
        c2.metric("Spam Messages", f"{(df['label']=='spam').sum():,}")
        c3.metric("Ham Messages",  f"{(df['label']=='ham').sum():,}")
        c4.metric("Spam Rate",     f"{(df['label']=='spam').mean()*100:.1f}%")
    except Exception:
        st.warning("Dataset not found. Place spam.csv in the G6_NLP_FP/data folder.")

# ──────────────────────
# PAGE 2 — SPAM DETECTOR
# ──────────────────────
elif page == "Spam Detector":
    st.title("Spam Detector")
    st.markdown("Paste any message below to check if it is spam or not.")
    st.markdown("---")

    try:
        bow, lr, nb = load_models()
    except Exception:
        st.error("Model files not found. Please run the Jupyter Notebook first to generate /models/.")
        st.stop()

    # Model selector
    model_choice = st.selectbox(
        "Choose a model:",
        ["Logistic Regression", "Naive Bayes"]
    )

    # Text input area
    # Read from session_state if an example button was clicked
    default_text = st.session_state.get('example_text', '')

    user_input = st.text_area(
        "Enter your message here:",
        value=default_text,
        height=150,
        placeholder="e.g. Congratulations! You won a FREE iPhone. Click now to claim..."
    )

    if st.button("Analyze", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter a message before clicking Analyze.")
        else:
            # Preprocess and vectorize
            cleaned = preprocess_text(user_input)
            vec     = bow.transform([cleaned])

            # Select model
            model = lr if "Logistic" in model_choice else nb

            # Predict
            pred  = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]

            spam_prob = proba[1]
            ham_prob  = proba[0]

            st.markdown("---")
            st.markdown("#### Result")

            if pred == 1:
                st.error(f"**SPAM** detected! (Confidence: {spam_prob:.1%})")
            else:
                st.success(f"**HAM** (legitimate) (Confidence: {ham_prob:.1%})")

            # Confidence bar chart
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(5, 2.5))
                bars = ax.barh(
                    ['Ham', 'Spam'],
                    [ham_prob, spam_prob],
                    color=['#2ecc71', '#e74c3c'],
                    edgecolor='black'
                )
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probability')
                ax.set_title('Prediction Confidence')
                for bar, val in zip(bars, [ham_prob, spam_prob]):
                    ax.text(
                        val + 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{val:.1%}', va='center'
                    )
                st.pyplot(fig)
                plt.close()

            with col2:
                st.markdown("**Preprocessed text:**")
                st.code(cleaned if cleaned else "(empty after preprocessing)")

                # Show top 5 spam-indicator words found in the message
                if hasattr(model, 'coef_'):
                    feature_names = np.array(bow.get_feature_names_out())
                    coef = model.coef_[0]
                    input_tokens = cleaned.split()
                    # Find which input tokens exist in the vocabulary
                    found = [(t, coef[np.where(feature_names == t)[0][0]])
                             for t in input_tokens
                             if t in feature_names]
                    if found:
                        found_sorted = sorted(found, key=lambda x: abs(x[1]), reverse=True)[:5]
                        st.markdown("**Key words that influenced the prediction:**")
                        for word, score in found_sorted:
                            direction = "spam" if score > 0 else "ham"
                            st.write(f"- `{word}` → {direction} (score: {score:.2f})")

    # Quick test examples — copy and paste into the text box above
    st.markdown("---")
    st.markdown("#### Try these examples (copy and paste into the text box above)")

    st.markdown("**Spam Example 1:**")
    st.code("WINNER!! You have been selected for a $1000 gift card. Call 0800-PRIZE now!")

    st.markdown("**Spam Example 2:**")
    st.code("FREE entry in 2 a weekly comp to win FA Cup final tkts 21st May 2005.")

    st.markdown("**Ham Example 1:**")
    st.code("Hey, are you coming to the study group tonight?")

    st.markdown("**Ham Example 2:**")
    st.code("Ok cool, I'll meet you at the library at 3pm.")

# ────────────────────────
# PAGE 3 — VISUALIZATIONS
# ────────────────────────
elif page == "Visualizations":
    st.title("Visualizations")
    st.markdown("---")

    figures_path = 'data/figures'
    fig_files = {
        'fig1_class_distribution.png': 'Figure 1 - Class Distribution',
        'fig2_wordcloud.png':          'Figure 2 - Word Cloud (Spam vs Ham)',
        'fig3_confusion_matrix.png':   'Figure 3 - Confusion Matrices',
        'fig4_model_comparison.png':   'Figure 4 - Model Comparison',
        'fig5_top_words.png':          'Figure 5 - Top Discriminative Words',
    }
    
    for fname, title in fig_files.items():
        fpath = os.path.join(figures_path, fname)
        if os.path.exists(fpath):
            st.markdown(f"### {title}")
            st.image(fpath)
            st.markdown("---")
        else:
            st.warning(f"{fname} not found - run the notebook first to generate figures.")
    
