from joblib import load
from preprocessing import preprocess

model = load('../models/spam_model.pkl')
vectorizer = load('../models/vectorizer.pkl')

def predict_spam(text):
    cleaned = preprocess(text)
    vectorized = vectorizer.transform([cleaned])
    result = model.predict(vectorized)
    return "Spam" if result[0] == 1 else "Not Spam"

# Example usage
if __name__ == "__main__":
    test_email = "Win money now!!! Click to claim your prize!"
    print("Prediction:", predict_spam(test_email))
