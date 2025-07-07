import joblib

model = joblib.load("model.pkl")
embedder = joblib.load("embedder.pkl")

test_input = "ordered lunch from swiggy"
vec = embedder.encode([test_input])
prediction = model.predict(vec)
print(prediction)
