import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Next Word Predictor", page_icon="🎬", layout="centered")

@st.cache_resource
def load_artifacts():
    model = load_model('next_word_lstm.h5')
    with open('tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('config.pickle', 'rb') as f:
        config = pickle.load(f)
    return model, tokenizer, config['max_len']

model, tokenizer, MAX_LEN = load_artifacts()

def predict_next_words(model, tokenizer, text, num_words, temperature):
    result = []
    current = text.strip().lower()
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([current])[0]
        token_list = token_list[-MAX_LEN:]
        token_list = pad_sequences([token_list], maxlen=MAX_LEN, padding='pre')
        probs = model.predict(token_list, verbose=0)[0]
        probs = np.log(probs + 1e-7) / temperature
        probs = np.exp(probs) / np.sum(np.exp(probs))
        idx = np.random.choice(len(probs), p=probs)
        word = tokenizer.index_word.get(idx, '')
        if not word:
            break
        result.append(word)
        current += ' ' + word
    return result, current

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Next Word Predictor")
st.caption("Cornell Movie Dialogues")
st.divider()

input_text  = st.text_input("Enter a sequence of words", placeholder="e.g. i don't know")
num_words   = st.slider("Words to predict", min_value=1, max_value=3, value=1)
temperature = st.slider(
    "Creativity",
    min_value=0.5, max_value=1.2, value=0.8, step=0.1,
    help="0.5 = safe/repetitive · 0.8 = balanced · 1.2 = creative/random"
)

if st.button("Predict", type="primary", use_container_width=True):
    if not input_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Predicting..."):
            words, full_text = predict_next_words(
                model, tokenizer, input_text, num_words, temperature
            )
        st.divider()
        st.markdown(f"**Predicted:** {' '.join(f'`{w}`' for w in words)}")
        st.markdown(f"**Full text:** _{full_text}_")

st.divider()
st.caption("Dataset: Cornell Movie Dialogues")
