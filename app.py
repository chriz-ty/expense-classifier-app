import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# -------------------------
st.set_page_config(page_title="Expense Classifier", layout="centered")

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    embedder = SentenceTransformer("embedder/")
    return model, embedder

model, embedder = load_model()
THRESHOLD = 0.2

if 'predictions' not in st.session_state:
    st.session_state.predictions = []

st.title("💸 Smart Expense Category Classifier")

# 🧱 Layout: Two columns
# Spacer for top margin
st.markdown("<div style='margin-top: -30px;'></div>", unsafe_allow_html=True)

# Define wider columns with spacing in between
left_col, spacer, right_col = st.columns([20.8, 0.8, 20.8], gap="large")

# -------------------------
# ✍️ Left Column: Manual Input
with left_col:
    st.markdown("<div style='text-align: left;'>", unsafe_allow_html=True)
    st.header("✍️ Manual Entry")

    user_input = st.text_input("🧾 Expense Line")

    if st.button("Classify Expense"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text.")
        else:
            vector = embedder.encode([user_input])
            proba = model.predict_proba(vector)[0] 
            prediction = model.predict(vector)[0]
            confidence = proba.max()
            classes = model.classes_

            if confidence < THRESHOLD or prediction == "Unknown":
                # Show Top 3 predictions
                top_indices = proba.argsort()[-3:][::-1]
                st.error("❌ Low confidence. Unable to classify confidently.")
                st.caption(f"(Top prediction: {prediction}, Confidence: {confidence:.2f})")

                st.info("🔍 Top 3 probable categories:")
                for i in top_indices:
                    st.write(f"• **{classes[i]}** — Confidence: `{proba[i]:.2f}`")
            else:
                st.success(f"✅ Predicted Category: **{prediction}**")
                st.caption(f"Confidence: {confidence:.2f}")
                st.session_state.predictions.append(prediction)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# 📤 Right Column: CSV Upload
with right_col:
    st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
    st.header("📁 Upload CSV for Bulk Prediction")

    uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if 'text' not in df.columns:
                st.error("❌ The CSV must contain a 'text' column.")
            else:
                st.success(f"✅ {len(df)} entries loaded!")

                vectors = embedder.encode(df["text"].astype(str).tolist())
                df["Predicted Category"] = model.predict(vectors)
                df["Confidence"] = model.predict_proba(vectors).max(axis=1)

                st.write("📊 Prediction Results:")
                st.dataframe(df[["text", "Predicted Category", "Confidence"]])

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results as CSV", csv, "predictions.csv", "text/csv")

                filtered_df = df[df["Predicted Category"] != "Unknown"]
                low_conf_df = df[(df["Predicted Category"] == "Unknown") | (df["Confidence"] < THRESHOLD)]

                if not low_conf_df.empty:
                    st.warning("⚠️ Some inputs were classified as Unknown or have low confidence.")
                    st.write(low_conf_df[["text", "Predicted Category", "Confidence"]])

                if not filtered_df.empty:
                    st.subheader("📊 Prediction Breakdown")
                    category_counts = filtered_df["Predicted Category"].value_counts()
                    fig, ax = plt.subplots()
                    ax.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
                    ax.axis("equal")
                    st.pyplot(fig)
                else:
                    st.info("No confident predictions to visualize.")
        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")
    st.markdown("</div>", unsafe_allow_html=True)
