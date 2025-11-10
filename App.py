import streamlit as st

# ✅ MUST be the first Streamlit command
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="wide",
)

# --- background image ---
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://i.postimg.cc/GptS6K5y/3816798.jpg");
    background-size: cover;              /* fills the whole screen */
    background-position: center;         /* centers the image */
    background-repeat: no-repeat;        /* no tiling */
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);           /* transparent header */
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- now the rest of your imports and setup ---
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer

# download resources after page_config
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

text_porter = PorterStemmer()
tfidf = pickle.load(open('./vectorizer.pkl', 'rb'))
classifier = pickle.load(open('./model.pkl', 'rb'))


# --- preprocessing function ---
def text_preprocessing(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    text = [i for i in y if i not in nltk.corpus.stopwords.words("english") and i not in string.punctuation]
    text = [text_porter.stem(i) for i in text]
    return " ".join(text)


# --- UI section ---
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
