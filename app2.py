import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from huggingface_hub import InferenceClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image

# ✅ Load .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# 🔐 Check token
if HF_TOKEN is None:
    st.error("❌ HF_TOKEN not found. Please check your .env file.")
    st.stop()

# ✅ Hugging Face clients
image_client = InferenceClient(provider="nebius", api_key=HF_TOKEN)
text_client = InferenceClient(provider="novita", api_key=HF_TOKEN)

# ✅ Streamlit page setup
st.set_page_config(page_title="🧠 Multi-AI Toolkit", layout="centered")
st.title("🧠Praveenraja,Anas,Ranjith-AI App: Image + Chat + PDF + CSV Analyzer")

# ✅ Tabs setup
tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️ Image Generator", 
    "💬 LLaMA Chat", 
    "📄 PDF Summarizer", 
    "📊 CSV Analyzer + Model"
])

# ----------------------
# 🖼️ Tab 1: Image Generator
# ----------------------
with tab1:
    st.header("🖼️ Generate Image from Text")
    prompt = st.text_input("🎨 Enter image prompt:", "Astronaut riding a horse")

    if st.button("🎨 Generate Image"):
        with st.spinner("🧠 Creating image..."):
            try:
                image = image_client.text_to_image(prompt=prompt, model="black-forest-labs/FLUX.1-dev")
                img_path = os.path.join(os.getcwd(), "generated_image.png")
                image.save(img_path)
                st.image(image, caption="Generated Image", use_container_width=True)
                with open(img_path, "rb") as f:
                    st.download_button("📥 Download Image", f, "generated_image.png", "image/png")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ----------------------
# 💬 Tab 2: LLaMA Chat
# ----------------------
with tab2:
    st.header("💬 Chat with LLaMA 3.1 8B")
    user_input = st.text_area("✍️ Ask a question:", placeholder="e.g., What is the capital of France?")

    if st.button("🚀 Get LLaMA Response"):
        if user_input.strip():
            with st.spinner("🤖 Thinking..."):
                try:
                    res = text_client.chat.completions.create(
                        model="meta-llama/Llama-3.1-8B-Instruct",
                        messages=[{"role": "user", "content": user_input}],
                    )
                    reply = res.choices[0].message["content"]
                    st.subheader("🧾 Response:")
                    st.write(reply)

                    with open("llama_output.txt", "w", encoding="utf-8") as f:
                        f.write(reply)
                    with open("llama_output.txt", "rb") as f:
                        st.download_button("📥 Download .txt", f, "llama_output.txt", "text/plain")
                except Exception as e:
                    st.error(f"❌ Chat error: {e}")
        else:
            st.warning("⚠️ Please enter a question.")

# ----------------------
# 📄 Tab 3: PDF Summarizer
# ----------------------
with tab3:
    st.header("📄 Summarize a PDF File")
    uploaded_pdf = st.file_uploader("📤 Upload a PDF file", type="pdf")

    if uploaded_pdf:
        try:
            reader = PdfReader(uploaded_pdf)
            pdf_text = ""
            for page in reader.pages:
                pdf_text += page.extract_text()

            st.subheader("📚 Preview (First 1000 characters):")
            st.text(pdf_text[:1000] + "...")

            if st.button("🧠 Summarize PDF"):
                with st.spinner("📝 Summarizing..."):
                    try:
                        summary_prompt = f"Summarize the following PDF content:\n{pdf_text[:3000]}"
                        res = text_client.chat.completions.create(
                            model="meta-llama/Llama-3.1-8B-Instruct",
                            messages=[{"role": "user", "content": summary_prompt}],
                        )
                        summary = res.choices[0].message["content"]

                        st.subheader("📝 Summary:")
                        st.write(summary)

                        with open("pdf_summary.txt", "w", encoding="utf-8") as f:
                            f.write(summary)
                        with open("pdf_summary.txt", "rb") as f:
                            st.download_button("📥 Download Summary", f, "pdf_summary.txt", "text/plain")
                    except Exception as e:
                        st.error(f"❌ Error summarizing: {e}")
        except Exception as e:
            st.error(f"❌ Could not read PDF: {e}")

# ----------------------
# 📊 Tab 4: CSV Analyzer + Model
# ----------------------
with tab4:
    st.header("📊 CSV Analyzer & Model Prediction")
    csv_file = st.file_uploader("📂 Upload a CSV file", type="csv")

    if csv_file:
        try:
            df = pd.read_csv(csv_file)
            st.subheader("🔍 Dataset Preview:")
            st.write(df.head())

            st.subheader("📈 Data Summary:")
            st.write(df.describe())
            st.write(f"🧾 Rows: {df.shape[0]}, Columns: {df.shape[1]}")

            st.subheader("📊 Correlation Heatmap:")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            st.subheader("🎯 Select Target Column for ML")
            target_column = st.selectbox("Choose column to predict:", df.columns)

            if target_column:
                X = df.drop(columns=[target_column])
                y = df[target_column]
                X = pd.get_dummies(X)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = RandomForestClassifier()
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                st.subheader("📈 Model Results")
                st.write(f"✅ Accuracy: **{acc * 100:.2f}%**")
                st.markdown(f"""
                **Model Summary**
                - 📊 Features used: {len(X.columns)}
                - 🎯 Target: `{target_column}`
                - 🤖 Model: RandomForestClassifier
                - ✅ Accuracy: `{acc * 100:.2f}%`
                """)
        except Exception as e:
            st.error(f"❌ Error processing CSV: {e}")
