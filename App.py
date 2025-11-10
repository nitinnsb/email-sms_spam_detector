import streamlit as st
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer

# ✅ First Streamlit command
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="wide",
)

# --- Custom CSS for visuals only ---
page_bg_img = """
<style>
/* Fullscreen background */
[data-testid="stAppViewContainer"] {
    background-image: url("https://i.postimg.cc/k5VDDrH9/Whats-App-Image-2025-11-10-at-15-39-05-c29ef7d9.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

/* Transparent header */
[data-testid="stHeader"], [data-testid="stToolbar"] {
    background: rgba(0,0,0,0);
}

/* Center content nicely */
.block-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
}

/* Glass box style */
.main-box {
    background: rgba(0, 0, 0, 0.6);
    padding: 2.5rem 3rem;
    border-radius: 1rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 25px rgba(0,0,0,0.5);
    text-align: center;
    width: 90%;
    max-width: 600px;
}

/* White readable text */
h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
    color: #ffffff !important;
}

/* Text input area */
textarea {
    background-color: rgba(255,255,255,0.1) !important;
    color: #ffffff !important;
    border: 1px solid #00ADB5 !important;
    border-radius: 10px !important;
}

/* Button styling */
button[kind="primary"] {
    background: linear-gradient(90deg, #00ADB5, #007B83) !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 0.6rem 1.5rem !important;
    transition: 0.3s ease;
}
button[kind="primary"]:hover {
    background: linear-gradient(90deg, #007B83, #00ADB5) !important;
    transform: scale(1.03);
}

/* Alert styling (Spam / Not Spam) */
.stAlert {
    background: rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    font-size: 1.1rem !important;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- NLTK setup ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- Model loading ---
text_porter = PorterStemmer()
tfidf = pickle.load(open('./vectorizer.pkl', 'rb'))
classifier = pickle.load(open('./model.pkl', 'rb'))

# --- Preprocessing ---
def text_preprocessing(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    text = [i for i in y if i not in nltk.corpus.stopwords.words("english") and i not in string.punctuation]
    text = [text_porter.stem(i) for i in text]
    return " ".join(text)

# --- UI section (same logic, new visuals) ---
st.markdown('<div class="main-box">', unsafe_allow_html=True)
st.title('📩 SMS Spam Classifier')
st.write("This app classifies messages as **Spam** or **Not Spam** based on their content.")

input_text = st.text_area('✉️ Enter your message here:')

if st.button('Predict'):
    if input_text.strip():
        processed = text_preprocessing(input_text)
        vectorized = tfidf.transform([processed])
        result = classifier.predict(vectorized)[0]
        if result == 1:
            st.success('🚨 **Spam Detected!**')
        else:
            st.info('✅ **Not Spam**')
    else:
        st.warning('⚠️ Please enter a message to classify.')
st.markdown('</div>', unsafe_allow_html=True)
