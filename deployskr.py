import streamlit as st
import numpy as np
import pickle
import html
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import time
import string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load Word2Vec model
w2v_model = Word2Vec.load('word2vec_job_posting.model')
vector_size = w2v_model.vector_size

# Load LSTM model
model = load_model('model_fake_job_detection.h5')

# Konfigurasi
max_len = 100
vocab_size = len(tokenizer.word_index) + 1

# Preprocessing input
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def preprocess_input(text):
    text = html.unescape(text) #Menghapus HTML Entities
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    text = text.replace('\xa0', ' ')
    text = re.sub(r'#URL_[a-f0-9]{64}#', '', text) #Menghapus Link URL
    text = re.sub(r'\burl[a-z0-9]{10,}\b', '', text) #Menghapus Link Token
    text = re.sub(r'[^\x00-\x7f]', '', text) # Hapus non-ASCII
    text = text.lower()
    text = re.sub(r'\.(\w)', r'. \1', text)
    text = re.sub(r'\r|\n', ' ', text) # Ganti newline dengan spasi
    text = re.sub(r'\d+', '', text) # Hapus angka
    text = ' '.join(word for word in text.split() if word not in stop_words) #Stopwords
    tokens = text.lower().split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens] #Lemmatization
    sequence = tokenizer.texts_to_sequences([tokens])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post')
    return padded


# Streamlit UI
st.title("🔍 Fake Job Posting Classifier")
st.write("Masukkan teks iklan lowongan kerja untuk diklasifikasikan.")

user_input = st.text_area("Input Teks", height=150, placeholder="Contoh: 'We are hiring remote data entry specialist with no experience needed...'")

if st.button("🚀 Prediksi", type="primary"):
    if user_input.strip() == "":
        st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")
    else:
        # Show loading spinner
        with st.spinner('🔄 Memproses teks dan melakukan prediksi...'):
            # Add a small delay for better UX (optional)
            time.sleep(0.5)
            
            # Process the input
            processed_input = preprocess_input(user_input)
            prediction = model.predict(processed_input)[0][0]

        # Determine label and emoji
        if prediction >= 0.5:
            label = "FAKE"
            emoji = "❌"
            prob = prediction
            color = "red"
        else:
            label = "REAL"
            emoji = "✅"
            prob = 1 - prediction
            color = "green"

        # Display results with enhanced styling
        st.divider()
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"<h1 style='text-align: center; margin: 0;'>{emoji}</h1>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"### Hasil Prediksi: <span style='color: {color}'>{label}</span>", unsafe_allow_html=True)
            st.markdown(f"**Probabilitas: {prob*100:.2f}%**")
        
        # Progress bar for probability
        st.progress(float(prob))
        
        # Additional info based on result
        if label == "FAKE":
            st.error("⚠️ Hati-hati! Iklan ini kemungkinan besar adalah PENIPUAN.")
        else:
            st.success("✅ Iklan ini kemungkinan LEGITIMATE/ASLI.")
