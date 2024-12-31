import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re
import plotly.graph_objects as go

# Ensure the NLTK data directory exists
nltk_data_dir = '/home/appuser/nltk_data'
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Set the NLTK data path
nltk.data.path.append(nltk_data_dir)

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Preprocessing Class
class Preprocessing:
    def __init__(self):
        self.sett = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = word_tokenize(text)
        text = [self.lemmatizer.lemmatize(word) for word in text if word not in self.sett]
        return ' '.join(text)

    def transform(self, X):
        return X.apply(self.clean_text)

# Load models and preprocessing tools
@st.cache_resource
def load_models():
    try:
        # Load models using TensorFlow format
        rnn_model = tf.keras.models.load_model('models/rnn_model')
        lstm_model = tf.keras.models.load_model('models/lstm_model')
        
        # Load preprocessor and tokenizer
        with open('models/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        with open('models/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
            
        # Load training histories
        with open('models/rnn_history.pkl', 'rb') as f:
            rnn_history = pickle.load(f)
        with open('models/lstm_history.pkl', 'rb') as f:
            lstm_history = pickle.load(f)
            
        return rnn_model, lstm_model, preprocessor, tokenizer, lstm_history, rnn_history
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None, None

def predict_text(text, model, preprocessor, tokenizer, max_len=200):
    processed_text = preprocessor.transform(pd.Series(text))
    text_seq = tokenizer.texts_to_sequences(processed_text)
    text_padded = pad_sequences(text_seq, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(text_padded)
    return 1 if prediction[0][0] > 0.5 else 0

def main():
    st.set_page_config(
        page_title="Fake News Classifier",
        page_icon="🗞️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # [Previous CSS styles remain the same]
    st.markdown("""
    <style>
    .main-title { font-size: 3rem; font-weight: bold; color: #2C3E50; text-align: center; margin-bottom: 30px; }
    .subtitle { font-size: 1.2rem; color: #34495E; text-align: center; margin-bottom: 20px; }
    .stTextArea textarea { border: 2px solid #3498DB; border-radius: 10px; }
    .stButton>button { background-color: #3498DB; color: white; font-weight: bold; border-radius: 10px; width: 100%; padding: 10px; }
    .stButton>button:hover { background-color: #2980B9; }
    .prediction-box { background-color: #F1F8E9; border-radius: 10px; padding: 20px; text-align: center; margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='main-title'>🗞️ Fake News Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Detect potential fake news using advanced machine learning models</p>", unsafe_allow_html=True)
    
    # Load all components
    rnn_model, lstm_model, preprocessor, tokenizer, lstm_history, rnn_history = load_models()
    
    if not all([rnn_model, lstm_model, preprocessor, tokenizer]):
        st.error("Unable to load models. Please ensure all model files are present.")
        return

    # Sidebar
    st.sidebar.header("🤖 Model Selection")
    model_choice = st.sidebar.radio(
        "Choose a Classification Model", 
        ["RNN", "LSTM"],
        index=1
    )
    
    # Main input area
    st.markdown("### Enter News Article Text")
    text_input = st.text_area(
        "Paste the text you want to classify", 
        height=250,
        placeholder="Enter the news article text here..."
    )
    
    # Classification
    if st.button("Classify News"):
        if not text_input:
            st.warning("Please enter some text to classify.")
            return
        
        model = rnn_model if model_choice == "RNN" else lstm_model
        model_name = "Recurrent Neural Network (RNN)" if model_choice == "RNN" else "Long Short-Term Memory (LSTM)"
        
        prediction = predict_text(text_input, model, preprocessor, tokenizer)
        
        st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
        if prediction == 1:
            st.error(f"🚨 Potential Fake News Detected (Model: {model_name})")
            st.markdown("""
            <p style='color:#E74C3C;'>
            The article shows characteristics of potential misinformation. 
            Please verify the source and cross-check with reliable news outlets.
            </p>
            """, unsafe_allow_html=True)
        else:
            st.success(f"✅ Appears to be Credible News (Model: {model_name})")
            st.markdown("""
            <p style='color:#2ECC71;'>
            The article seems to have characteristics of legitimate news. 
            However, always maintain a critical perspective.
            </p>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Training history visualization
    if all([lstm_history, rnn_history]):
        selected_model = "LSTM" if model_choice == "LSTM" else "RNN"
        
        history = lstm_history if selected_model == "LSTM" else rnn_history
        
        # Create subplots
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'{selected_model} Model Loss', f'{selected_model} Model Accuracy'),
            vertical_spacing=0.15
        )
        
        # Add loss traces
        fig.add_trace(
            go.Scatter(y=history['loss'], name='Training Loss', line=dict(color='#2ecc71')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=history['val_loss'], name='Validation Loss', line=dict(color='#e74c3c')),
            row=1, col=1
        )
        
        # Add accuracy traces
        fig.add_trace(
            go.Scatter(y=history['accuracy'], name='Training Accuracy', line=dict(color='#2ecc71')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(y=history['val_accuracy'], name='Validation Accuracy', line=dict(color='#e74c3c')),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            showlegend=True,
            template="plotly_dark",
            title_text=f"{selected_model} Model Performance",
            title_x=0.5
        )
        
        # Update y-axes labels
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2)
        
        # Update x-axes labels
        fig.update_xaxes(title_text="Epochs", row=1, col=1)
        fig.update_xaxes(title_text="Epochs", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # About section
    st.markdown("---")
    with st.expander("ℹ️ About the Model"):
        st.markdown("""
        ### How does this Fake News Classifier work?
        
        - We use advanced deep learning models (RNN and LSTM) to analyze text
        - The models are trained on a large dataset of news articles
        - Text is preprocessed by:
          * Removing punctuation and numbers
          * Lemmatizing words
          * Removing stop words
        - The model predicts whether a given text is likely to be fake or credible
        
        ### Limitations
        - No AI model is 100% accurate
        - Always verify news from multiple sources
        - This is a supportive tool, not a definitive fact-checker
        """)

if __name__ == "__main__":
    main()