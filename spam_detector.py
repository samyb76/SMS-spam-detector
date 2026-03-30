import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[["v1", "v2"]]
df.columns = ["label", "message"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})

print(
    f"Dataset : {len(df)} messages ({df['label'].sum()} spam, {(df['label']==0).sum()} ham)"
)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["message"] = df["message"].apply(clean_text)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(df["message"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n── Résultats ──────────────────────────────────────────")
print(f"Accuracy : {model.score(X_test, y_test):.4f}")
print("\nClassification Report :")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
print("Confusion Matrix :")
print(confusion_matrix(y_test, y_pred))

with open("spam_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)
print("\nModèle sauvegardé dans spam_model.pkl")


def predict_spam(message):
    message = clean_text(message)
    vect = vectorizer.transform([message])
    result = model.predict(vect)
    proba = model.predict_proba(vect)[0]
    label = "SPAM" if result[0] == 1 else "PAS SPAM"
    confidence = proba[1] if result[0] == 1 else proba[0]
    return f"{label} (confiance : {confidence:.1%})"


print("\n── Tests ──────────────────────────────────────────────")
test_messages = [
    "Congratulations you won a free prize call now",
    "Hey are we still meeting tomorrow for lunch?",
    "URGENT: Your account has been compromised click here",
    "Ok sounds good see you later",
]

for msg in test_messages:
    print(f"'{msg}'\n→ {predict_spam(msg)}\n")
