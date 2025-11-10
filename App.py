import streamlit as st
import pickle
import string
import nltk
nltk.download('punkt_tab')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
text_porter = PorterStemmer()
tfidf = pickle.load(open('./vectorizer.pkl', 'rb'))
classifier = pickle.load(open('./model.pkl', 'rb'))

st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="wide",
)

def text_preprocessing(text):
    text =  text.lower() # make all characters in lower case
    text = nltk.word_tokenize(text) # make a list of all words
    y = []
    
    for i in text:    # keeping only alpha numeric characters
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    
    for i in text:
        if i not in nltk.corpus.stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(text_porter.stem(i))

    return " ".join(y)



st.title('SMS Spam Classifier')
st.write("This application classifies messages as 'Spam' or 'Not Spam' based on text content.")

input = st.text_area('Enter the message')

if st.button('Predict'):
    if input:
        # Preproccess input
        input = text_preprocessing(input)

        # text vectorization
        input = tfidf.transform([input])

        # predict
        result = classifier.predict(input)[0]

        # display
        if result == 1:
            st.header('Spam')
        else:
            st.header('Not Spam')
    else:
        st.write('Please enter a message to classify')
