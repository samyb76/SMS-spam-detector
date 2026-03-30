import pandas as pd
import re
import pickle
from collections import Counter

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
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
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_data
def load_and_train(path):
    df = pd.read_csv(path, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "message"]
    df["label_num"] = (df["label"] == "spam").astype(int)
    df["length"] = df["message"].str.len()
    df["word_count"] = df["message"].str.split().str.len()
    df["clean"] = df["message"].apply(clean_text)

    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vec.fit_transform(df["clean"])
    y = df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return df, model, vec, report, cm


st.sidebar.title("📊 SMS Spam Detector")
uploaded = st.sidebar.file_uploader("Upload spam.csv", type="csv")
st.sidebar.markdown("---")

if uploaded is None:
    st.info("👈 Upload your **spam.csv** file in the sidebar to launch the dashboard.")
    st.stop()

df, model, vec, report, cm = load_and_train(uploaded)

n_total = len(df)
n_ham = (df["label"] == "ham").sum()
n_spam = (df["label"] == "spam").sum()
acc = round(report["accuracy"] * 100, 2)

st.title("📊 SMS Spam Detector — Dashboard")
st.markdown("---")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Messages", f"{n_total:,}")
c2.metric("Ham", f"{n_ham:,}", f"{n_ham/n_total*100:.1f}%")
c3.metric("Spam", f"{n_spam:,}", f"{n_spam/n_total*100:.1f}%")
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

with col3:

    STOPWORDS = set(
        "to the a of and in is it you your for on are be i this have that was "
        "with at he from they we as an do by his or all but not what so can out "
        "if up then there their about more will when no just one s t u r n m d p c".split()
    )

with col3:
    st.subheader("Top 10 Words in Spam Messages")
    spam_words = []
    for msg in df[df.label == "spam"]["message"]:
        words = re.findall(r"[a-zA-Z]{3,}", msg.lower())
        spam_words += [w for w in words if w not in STOPWORDS]
    top_words = Counter(spam_words).most_common(10)
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

st.markdown("---")

st.subheader("🔍 Test a Message")
user_msg = st.text_area(
    "Paste an SMS here:", placeholder="Ex: Congratulations! You've won a free prize..."
)
if st.button("Analyze"):
    if user_msg.strip():
        cleaned = clean_text(user_msg)
        vect = vec.transform([cleaned])
        pred = model.predict(vect)[0]
        proba = model.predict_proba(vect)[0]
        label = "🚨 SPAM" if pred == 1 else "✅ Ham"
        conf = proba[pred] * 100
        color = SPAM_COLOR if pred == 1 else HAM_COLOR
        st.markdown(
            f"<h3 style='color:{color}'>{label} — Confidence: {conf:.1f}%</h3>",
            unsafe_allow_html=True,
        )
        st.progress(int(conf))
    else:
        st.warning("Please enter a message first!")
