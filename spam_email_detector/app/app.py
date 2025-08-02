import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import preprocess
from joblib import load

model = load('models/spam_model.pkl')
vectorizer = load('models/vectorizer.pkl')

st.title("📧 Spam Email Detector")
st.markdown("Paste any email message below to predict whether it's spam or not.")

# Initialize session state for email input
if 'email' not in st.session_state:
    st.session_state.email = ''

# Function to clear the input
def clear_text():
    st.session_state.email = ''

# Text area bound to session state variable
email_input = st.text_area("Enter email content:", value=st.session_state.email, height=150, key='email')

# Buttons side by side
col1, col2 = st.columns(2)

with col1:
    if st.button("Clear"):
        clear_text()

with col2:
    if st.button("Predict"):
        if not st.session_state.email.strip():
            st.warning("⚠️ Please enter some email content to predict.")
        else:
            cleaned = preprocess(st.session_state.email)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            probabilities = model.predict_proba(vectorized)[0]

            st.markdown("### Cleaned (Preprocessed) Text:")
            st.write(cleaned)

            if prediction == 1:
                st.error(f"🚨 This email is **SPAM** with confidence {probabilities[1]*100:.2f}%.")
            else:
                st.success(f"✅ This email is **NOT SPAM** with confidence {probabilities[0]*100:.2f}%.")

# Show example spam emails at the bottom
st.markdown("---")
st.markdown("### Example Spam Emails to Try:")
example_spams = [
    "Congratulations! You've won a free trip to Bahamas! Click here to claim.",
    "Urgent! Your account has been compromised. Reset your password now.",
    "Get cheap meds online without prescription. Limited offer!",
    "You have won $1,000,000! Contact us immediately to claim your prize."
]

for i, spam in enumerate(example_spams, 1):
    st.markdown(f"{i}. {spam}")
