import pandas as pd
import re
from collections import Counter

import streamlit as st
import matplotlib.pyplot as plt

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(
    page_title="SMS Spam Detector — Dashboard",
    page_icon="📊",
    layout="wide",
)

HAM_COLOR = "#1D9E75"
SPAM_COLOR = "#D85A30"
TICK_COLOR = "#555555"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def suspicious_score(text):
    text = str(text).lower()

    patterns = [
        r"you[\' ]?ve won",
        r"claim (your )?(prize|reward)",
        r"free prize",
        r"winner",
        r"urgent",
        r"click here",
        r"verify your account",
        r"send me (all )?your information",
        r"personal information",
        r"bank details",
        r"credit card",
        r"limited time",
        r"congratulations",
        r"free smartphone",
        r"won a smartphone",
        r"send your details",
    ]

    score = 0
    for pattern in patterns:
        if re.search(pattern, text):
            score += 1

    return score


@st.cache_data
def load_and_train(path):
    df = pd.read_csv(path, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "message"]

    df["label_num"] = (df["label"] == "spam").astype(int)
    df["length"] = df["message"].astype(str).str.len()
    df["word_count"] = df["message"].astype(str).str.split().str.len()
    df["clean"] = df["message"].apply(clean_text)

    y = df["label_num"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean"],
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    word_vec = TfidfVectorizer(
        stop_words="english",
        max_features=8000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=5000,
        sublinear_tf=True,
    )

    X_train_word = word_vec.fit_transform(X_train_text)
    X_test_word = word_vec.transform(X_test_text)

    X_train_char = char_vec.fit_transform(X_train_text)
    X_test_char = char_vec.transform(X_test_text)

    X_train = hstack([X_train_word, X_train_char])
    X_test = hstack([X_test_word, X_test_char])

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return df, model, word_vec, char_vec, report, cm


st.sidebar.title("📊 SMS Spam Detector")
uploaded = st.sidebar.file_uploader("Upload spam.csv", type="csv")
st.sidebar.markdown("---")

if uploaded is None:
    st.info("👈 Upload your **spam.csv** file in the sidebar to launch the dashboard.")
    st.stop()

df, model, word_vec, char_vec, report, cm = load_and_train(uploaded)

n_total = len(df)
n_ham = (df["label"] == "ham").sum()
n_spam = (df["label"] == "spam").sum()
acc = round(report["accuracy"] * 100, 2)

st.title("📊 SMS Spam Detector — Dashboard")
st.markdown("---")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Messages", f"{n_total:,}")
c2.metric("Ham", f"{n_ham:,}", f"{n_ham / n_total * 100:.1f}%")
c3.metric("Spam", f"{n_spam:,}", f"{n_spam / n_total * 100:.1f}%")
c4.metric("Accuracy", f"{acc}%")

st.markdown("---")

col1, _ = st.columns(2)

with col1:
    st.subheader("Ham vs Spam Distribution")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        [n_ham, n_spam],
        labels=["Ham", "Spam"],
        colors=[HAM_COLOR, SPAM_COLOR],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"linewidth": 0},
        textprops={"fontsize": 12, "color": TICK_COLOR},
    )
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    st.pyplot(fig)
    plt.close()

st.markdown("---")

col3, _ = st.columns(2)

STOPWORDS = set(
    "to the a of and in is it you your for on are be i this have that was "
    "with at he from they we as an do by his or all but not what so can out "
    "if up then there their about more will when no just one s t u r n m d p c".split()
)

with col3:
    st.subheader("Top 10 Words in Spam Messages")
    spam_words = []

    for msg in df[df["label"] == "spam"]["message"]:
        words = re.findall(r"[a-zA-Z]{3,}", str(msg).lower())
        spam_words += [w for w in words if w not in STOPWORDS]

    top_words = Counter(spam_words).most_common(10)

    if top_words:
        words, counts = zip(*top_words)

        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.barh(words[::-1], counts[::-1], color=SPAM_COLOR, edgecolor="none")
        ax.bar_label(bars, padding=4, fontsize=10, color=TICK_COLOR)
        ax.set_xlabel("Occurrences", color=TICK_COLOR)
        ax.tick_params(colors=TICK_COLOR)

        for spine in ax.spines.values():
            spine.set_edgecolor(TICK_COLOR)

        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(left=False)
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        st.pyplot(fig)
        plt.close()
    else:
        st.warning("No spam words found in the dataset.")

st.markdown("---")

st.subheader("🔍 Test a Message")
user_msg = st.text_area(
    "Paste an SMS here:",
    placeholder="Example: Congratulations! You've won a free prize...",
)

if st.button("Analyze"):
    if user_msg.strip():
        cleaned = clean_text(user_msg)

        X_word = word_vec.transform([cleaned])
        X_char = char_vec.transform([cleaned])
        X_input = hstack([X_word, X_char])

        proba = model.predict_proba(X_input)[0]
        ml_spam_prob = proba[1] * 100

        rule_score = suspicious_score(user_msg)

        final_spam_score = ml_spam_prob
        if rule_score >= 1:
            final_spam_score += 12
        if rule_score >= 2:
            final_spam_score += 10
        if rule_score >= 3:
            final_spam_score += 10

        final_spam_score = min(final_spam_score, 99.0)

        pred = 1 if final_spam_score >= 50 else 0

        label = "🚨 SPAM" if pred == 1 else "✅ Ham"
        conf = final_spam_score if pred == 1 else 100 - final_spam_score
        color = SPAM_COLOR if pred == 1 else HAM_COLOR

        st.markdown(
            f"<h3 style='color:{color}'>{label} — Confidence: {conf:.1f}%</h3>",
            unsafe_allow_html=True,
        )

        st.progress(int(conf))

        st.caption(
            f"ML spam probability: {ml_spam_prob:.1f}% | Suspicious pattern score: {rule_score}"
        )
    else:
        st.warning("Please enter a message first!")
