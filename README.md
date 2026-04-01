# SMS Spam Detector — Dashboard

A machine learning-powered SMS spam detection dashboard built with Python, Streamlit, and scikit-learn. Upload your dataset, explore statistics, and test messages in real time.

---

## Features

* Upload any SMS dataset in `.csv` format directly from the sidebar
* Visual dashboard with key metrics: total messages, ham/spam counts, and model accuracy
* Pie chart showing **Ham vs Spam distribution**
* Bar chart of **Top 10 most frequent spam words**
* Real-time **message tester** — paste any SMS and get an instant prediction with confidence score
* Hybrid scoring system combining **ML probability** (TF-IDF + Logistic Regression) and **rule-based pattern matching**

---

## Technologies Used

* Python
* [Streamlit](https://streamlit.io/)
* [scikit-learn](https://scikit-learn.org/) — TF-IDF Vectorizer, Logistic Regression
* [pandas](https://pandas.pydata.org/)
* [matplotlib](https://matplotlib.org/)
* [scipy](https://scipy.org/) — sparse matrix support

---

## Prerequisites

### Dataset Format

The app expects a CSV file with at least two columns:

| Column | Description |
|--------|-------------|
| `v1` | Label — either `ham` or `spam` |
| `v2` | The raw SMS message text |

> The classic [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) from UCI / Kaggle works out of the box.

---

## Installation

1. Clone the repository

```bash
git clone https://github.com/your-username/sms-spam-detector.git
cd sms-spam-detector
```

2. Install the required libraries

```bash
pip install streamlit pandas scikit-learn matplotlib scipy
```

---

## Usage

Run the dashboard:

```bash
streamlit run spam_detector_dashboard.py
```

Then:

1. Open the browser at `http://localhost:8501`
2. In the **sidebar**, upload your `spam.csv` file
3. Explore the dashboard — charts and metrics load automatically
4. Scroll down to **"Test a Message"**, paste any SMS, and click **Analyze**

### Example

```
Congratulations! You've won a free prize. Click here to claim your reward now.
→ 🚨 SPAM — Confidence: 97.3%
```

```
Hey, are you coming to the meeting tomorrow at 10am?
→ ✅ Ham — Confidence: 99.1%
```

---

## How It Works

The model uses a **dual TF-IDF vectorization** approach:

* **Word-level TF-IDF** (1–2 grams, 8,000 features) captures common spam phrases
* **Character-level TF-IDF** (3–5 char n-grams, 5,000 features) catches obfuscated words like `fr33` or `pr1ze`

Both feature sets are combined with `hstack` and fed into a **Logistic Regression** classifier with balanced class weights.

On top of the ML score, a **rule-based pattern scorer** checks for suspicious phrases (e.g. *"you've won"*, *"click here"*, *"bank details"*) and boosts the spam probability accordingly.

> The model is trained fresh each time you upload a new dataset. Results may vary depending on dataset size and quality.

---

## Author

Samy BOUSSAD
