# train_and_save.py
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

# Load updated dataset
df = pd.read_csv("expense_data.csv")
X = df["text"]
y = df["category"]

# Encode text with BERT
embedder = SentenceTransformer('all-MiniLM-L6-v2')
X_vec = embedder.encode(X.tolist())

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

# Save model + encoder
joblib.dump(model, "model.pkl")
joblib.dump(embedder, "embedder.pkl")

print("✅ Model and embedder saved!")
