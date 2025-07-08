# 💸 Smart Expense Classifier

This is a **simple and smart app** that helps you **automatically identify what category your expense belongs to**, like `Food`, `Medical`, `Shopping`, etc. You can either type in your expense (like "Ordered dinner from Swiggy") or upload a list of expenses using a CSV file — and the app will tell you what each one is related to.
<br>
This project uses **Machine Learning and Natural Language Processing (NLP)** to make predictions. It was built with Python and Streamlit.

---

## 🧠 What It Does

- You type in a sentence like:  
  `Paid 250 for electricity bill`  
  👉 It predicts: **Utilities**

- Or upload a CSV file with many expenses like:  
`text
Ordered pizza from Domino's <br>
Booked movie tickets <br>
Bought a kurti from FabIndia`


👉 It predicts the category for each row like: `Food`, `Entertainment`, `Shopping`, etc.

---

## 🌟 Features

- ✅ Manual input prediction
- 📤 Upload CSV file and get batch predictions
- 📈 Shows a pie chart of expense categories (when uploading CSV)
- 💾 Download the results as a CSV
- 🚫 Shows a warning if the model is unsure (low confidence)

---

## 📸 Screenshot

![App Screenshot](link-to-screenshot-if-you-have-one)

---

## 🚀 Try the App Live

👉 **Hosted here**: [smart-expense-classifier.streamlit.app](https://expense-classifier-app-bcxkybcuceyxv5zpreuwnu.streamlit.app/)

---

## 🛠 How It Works (In Simple Terms)

1. **Model Training**  
 The app uses a pre-trained machine learning model. It was trained on a dataset of different kinds of expenses and their categories.

2. **Sentence Embedding**  
 It uses something called a “sentence transformer” to understand the meaning of each sentence and convert it into numbers.

3. **Prediction**  
 The trained model uses these numbers to predict the most likely category (like `Food` or `Medical`).

4. **Confidence Score**  
 If the model is not confident enough, it shows a warning and lists the top 3 possible categories.

---

## 📦 How to Run the App Locally

### 🔧 Prerequisites

- Python 3.8+
- Git (optional)
- pip

### 🧪 Setup Steps

1. Clone this repo:
 ```
 git clone https://github.com/chriz-ty/expense-classifier-app.git
 cd expense-classifier-app
```
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
    streamlit run app.py
   ```
4. The app will open in your browser.

### Folder Structure
```
expense-classifier-app/
├── app.py                # Main Streamlit app
├── model.pkl             # Trained classification model
├── embedder.pkl          # Sentence embedding model
├── expense_data.csv      # (Optional) Sample dataset
├── requirements.txt      # List of required Python libraries
└── README.md             # Project documentation
```
## Why I built this
I wanted to work on a beginner-friendly real-world project that uses NLP and machine learning — something practical like tracking daily expenses and automatically classifying them. This helped me understand model training, preprocessing, and deployment in a hands-on way.

## 🧑‍💻 Built With
Streamlit
scikit-learn
pandas  
sentence-transformers

## Contact
Christy Maria Sebastian <br>
LinkedIn: https://www.linkedin.com/in/christymariasebastian/
  
