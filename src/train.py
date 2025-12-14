import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
try:
    df = pd.read_csv("data/spam.csv", encoding="utf-8")
except:
    df = pd.read_csv("data/spam.csv", encoding="latin-1")

# Rename columns
df.columns = [c.lower() for c in df.columns]
df = df.rename(columns={"v1": "label", "v2": "text"})
df = df[["label", "text"]]

# Encode labels
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# Build pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
    ("nb", MultinomialNB())
])

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model/spam_classifier.pkl")
print("\nModel saved to model/spam_classifier.pkl")
