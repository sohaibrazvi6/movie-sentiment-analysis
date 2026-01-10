import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# ==========================================
# 2. LOAD SAVED MODELS
# ==========================================
# We use @st.cache_resource so it loads only once (faster)
@st.cache_resource
def load_lstm_resources():
    try:
        model = load_model('sentiment_model.keras')
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading LSTM files: {e}. Did you download them?")
        return None, None

@st.cache_resource
def load_lr_pipeline():
    try:
        with open('logistic_pipeline.pkl', 'rb') as f:
            pipeline = pickle.load(f)
        return pipeline
    except Exception as e:
        st.error(f"Error loading Logistic Regression file: {e}")
        return None

# ==========================================
# 3. SIDEBAR: CHOOSE YOUR MODEL
# ==========================================
st.sidebar.title("⚙️ Model Settings")
st.sidebar.write("Choose which AI brain to use:")

model_choice = st.sidebar.radio(
    "Select Model:",
    ("Logistic Regression (Recommended)", "Bidirectional LSTM (Deep Learning)")
)

st.sidebar.markdown("---")
if "Logistic" in model_choice:
    st.sidebar.success("✅ **Active: Logistic Regression**")
    st.sidebar.info("Uses Bigram analysis (understanding 2-word phrases). Currently simpler but highly accurate (~88%).")
else:
    st.sidebar.warning("🧠 **Active: Deep Learning**")
    st.sidebar.info("Uses a Neural Network that reads text forwards and backwards. Good for complex context (~86%).")

# ==========================================
# 4. MAIN APPLICATION UI
# ==========================================
st.title("🎬 Movie Review Analyzer")
st.write("Enter a movie review below to see if the sentiment is **Positive** or **Negative**.")

# Text Input Area
user_input = st.text_area("Type your review here:", height=150, placeholder="e.g., The visuals were stunning, but the plot made zero sense.")

# ==========================================
# 5. PREDICTION LOGIC
# ==========================================
if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please type a review first!")
    else:
        with st.spinner('Analyzing...'):
            try:
                # --- OPTION A: LOGISTIC REGRESSION ---
                if "Logistic" in model_choice:
                    pipeline = load_lr_pipeline()
                    if pipeline:
                        # 1. Predict (0 or 1)
                        prediction = pipeline.predict([user_input])[0]
                        # 2. Get Confidence Score
                        probs = pipeline.predict_proba([user_input])
                        confidence = np.max(probs) # Takes the higher probability
                        
                        is_positive = (prediction == 1)

                # --- OPTION B: LSTM (DEEP LEARNING) ---
                else:
                    lstm_model, tokenizer = load_lstm_resources()
                    if lstm_model and tokenizer:
                        # 1. Tokenize & Pad (Preprocess)
                        seq = tokenizer.texts_to_sequences([user_input])
                        padded = pad_sequences(seq, maxlen=200)
                        
                        # 2. Predict
                        score = lstm_model.predict(padded)[0][0]
                        
                        # 3. Interpret
                        is_positive = (score > 0.5)
                        confidence = score if is_positive else (1 - score)

                # --- DISPLAY RESULTS ---
                st.markdown("---")
                if is_positive:
                    st.success(f"### Result: POSITIVE Review 😊")
                    st.progress(int(confidence * 100))
                    st.write(f"**Confidence Score:** {confidence:.2%}")
                else:
                    st.error(f"### Result: NEGATIVE Review 😠")
                    st.progress(int(confidence * 100))
                    st.write(f"**Confidence Score:** {confidence:.2%}")

            except Exception as e:
                st.error(f"Something went wrong: {e}")

# Footer
st.markdown("---")
st.caption("Built with TensorFlow, Scikit-Learn, and Streamlit")