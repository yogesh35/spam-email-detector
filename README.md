# 📧 Spam Email Detector

A Python-based machine learning project to detect whether an email or SMS message is **Spam** or **Not Spam** using Natural Language Processing (NLP) and a Naive Bayes classifier.

---

## 🚀 Features

- Preprocesses text by cleaning, removing stopwords, and stemming
- Converts text into TF-IDF vectors for machine learning
- Trains a Multinomial Naive Bayes classifier achieving ~97.6% accuracy
- Provides both command-line and interactive Streamlit web app interfaces
- Displays prediction confidence and cleaned text for transparency
- Includes example spam emails for quick testing

---

## 📁 Project Structure

spam_email_detector/
├── data/
│   └── spam.csv               # Dataset (SMS Spam Collection)
├── src/
│   ├── preprocessing.py       # Text cleaning and preprocessing functions
│   ├── train_model.py         # Script to train and save the ML model
│   └── predict.py             # Command-line spam prediction script
├── models/
│   ├── spam_model.pkl         # Saved trained model
│   └── vectorizer.pkl         # Saved TF-IDF vectorizer
├── app/
│   └── app.py                 # Streamlit web application
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation



---

## 🛠️ Installation & Setup

1. **Clone the repository**

    ```
    git clone https://github.com/YOUR_USERNAME/spam-email-detector.git
    cd spam-email-detector
    ```

2. **Create a virtual environment (optional but recommended)**

    ```
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3. **Install dependencies**

    ```
    pip install -r requirements.txt
    ```

4. **Download dataset**

    Download the SMS Spam Collection Dataset and place the spam.csv file inside the data/ folder.

---

## 🧠 Training the Model

From the project root directory:

cd src
python train_model.py

This will preprocess the dataset, train the Naive Bayes model, and save the model and TF-IDF vectorizer files in the models/ folder.

---

## 🧪 Predicting Spam via Command Line

You can test any message by running:

python predict.py

Then enter your message when prompted. The program will output whether the message is spam or not.

---

## 🌐 Using the Streamlit Web App

Run the interactive web app with:

streamlit run app/app.py

- Paste your email or SMS message in the text area.
- Click Predict to see if it’s spam or not, along with the cleaned text and prediction confidence.
- Use the Clear button to reset input.
- Example spam messages are shown below the input for your convenience.

---

## 🧾 How It Works (Technical Summary)

- **Preprocessing:** Uses NLTK to remove stopwords and stem words for uniformity.
- **Feature Extraction:** Applies TF-IDF vectorization to convert text into numeric features.
- **Model:** Multinomial Naive Bayes classifier trained on the SMS Spam Collection Dataset.
- **Evaluation:** Achieves approximately 97.6% accuracy on test data.
- **Deployment:** Available as a CLI tool and a web app via Streamlit.

---

## 📊 Model Performance

| Metric      | Score     |
|-------------|-----------|
| Accuracy    | ~97.6%    |
| Precision   | High      |
| Recall      | High      |
| F1-Score    | High      |

(Detailed metrics can be added after evaluation)

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Improve the preprocessing pipeline
- Try other ML algorithms like SVM or Logistic Regression
- Add more features to the web app
- Fix bugs or improve documentation

Please fork the repo and submit pull requests.

---

## 📜 License

This project is licensed under the MIT License — see the LICENSE file for details.

---

## 👨‍💻 Author

**Yogesh R**

- Email: yogesh200412@gmail.com
- Domain: Cybersecurity enthusiast and Python developer

---

## 🔗 Useful Links

- Kaggle SMS Spam Collection Dataset
- Streamlit Documentation
- NLTK Documentation

Thank you for checking out this project! Feel free to reach out for any questions or collaboration.
