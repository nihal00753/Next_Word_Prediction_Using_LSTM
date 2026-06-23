# Next Word Predictor (LSTM)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Stacked%20LSTM-d00000)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Train%2FTest%20Split-f7931e)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![Trained On](https://img.shields.io/badge/Trained%20On-Kaggle%20P100%20GPU-20beff)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A sequence model that predicts the next word(s) in a sentence, trained on Cornell Movie Dialogues using a stacked LSTM network. Supports predicting 1–3 next words with adjustable temperature sampling for creative control.

## Overview

- Trained on ~60k lines of natural movie dialogue (Cornell Movie Dialogues corpus)
- Predicts the next 1–3 words given any input sequence
- Uses temperature sampling instead of argmax — avoids degenerate "the the the" predictions
- Key engineering decision: `sparse_categorical_crossentropy` instead of one-hot labels, cutting label RAM from 1.6 GB → 0.2 MB — the fix that made training possible without a data generator
- Built to demonstrate LSTM sequence modeling, tokenization pipelines, and Keras callbacks on a real NLP task


## Tech Stack

- **Modeling:** TensorFlow, Keras
- **Preprocessing:** scikit-learn (train/test split), Keras Tokenizer
- **Data:** Cornell Movie Dialogues (NLTK / Kaggle)
- **App:** Streamlit
- **Training:** Kaggle (Tesla P100 16GB GPU)

## Dataset

- **Source:** Cornell Movie Dialogues Corpus — 304,713 lines of movie script dialogue
- **Subset used:** 20% (~60k lines) — short conversational sentences, avg ~9 words per line
- **Why Cornell:** Movie dialogue lines are short and conversational. Short lines generate fewer n-gram sequences per line, avoiding the RAM explosion that kills training on prose datasets like Moby Dick
- **Vocabulary:** Top 5,000 words (Keras Tokenizer with `num_words=5000`)
- **OOV token:** `<OOV>` for unseen words at inference time

## Pipeline

1. Load `movie_lines.txt`, extract dialogue text field (5th `+++$+++` field)
2. Shuffle and take 20% subset
3. Clean text — lowercase, remove non-alphabetic characters, collapse whitespace
4. Drop lines with fewer than 3 words
5. Fit Keras `Tokenizer(num_words=5000)` on cleaned lines
6. Generate n-gram sequences, cap each to `MAX_LEN=15` tokens
7. Pad sequences with `padding='pre'`
8. Split into `X` (context) and `y` (next word index — sparse integer, not one-hot)
9. 90/10 train/val split (`random_state=42`)
10. Train with EarlyStopping + ReduceLROnPlateau
11. Save model + tokenizer + config for Streamlit inference

## Model Architecture

| Layer | Config | Output Shape |
|---|---|---|
| Embedding | vocab=5000, dim=128 | (batch, 15, 128) |
| LSTM | 128 units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2 | (batch, 15, 128) |
| LSTM | 64 units, dropout=0.2, recurrent_dropout=0.2 | (batch, 64) |
| Dropout | 0.3 | (batch, 64) |
| Dense | 128, ReLU | (batch, 128) |
| Dropout | 0.2 | (batch, 128) |
| Dense (Output) | 5000, Softmax | (batch, 5000) |

- **Loss:** `sparse_categorical_crossentropy` (integer labels, no one-hot)
- **Optimizer:** Adam (lr=1e-3)
- **Callbacks:** EarlyStopping (monitor=`val_loss`, patience=5, restore best weights), ReduceLROnPlateau (factor=0.5, patience=2, min_lr=1e-6)
- **Max epochs:** 50 (early stopping fired at epoch 27)

## Results

| Metric | Train | Validation |
|---|---|---|
| Accuracy | ~16.5% | ~17.0% |
| Stopped at epoch | 27 | — |
| Training platform | Kaggle P100 GPU | — |

**On accuracy:** 17% on a 5,000-word vocabulary is meaningful. Random chance = 1/5000 = 0.02%. The model is ~850× better than random. Next-word prediction accuracy is always low on large vocabularies — the real quality signal is whether predictions are linguistically coherent, not the raw number.

Train and val accuracy tracked closely throughout all 27 epochs with minimal divergence — the recurrent dropout inside both LSTM layers handled regularisation effectively.

![Training Curves](training_curves.png)

## Sample Predictions (temperature=0.8)

| Input | Predicted next 3 words |
|---|---|
| i don't know | what you mean |
| what are you | doing here now |
| we have to | get out of |
| i can't believe | you did that |

## Temperature Sampling

Instead of always picking the highest-probability word (argmax), the app uses temperature sampling:

- **0.5** — conservative, more repetitive but coherent
- **0.8** — balanced (recommended default)
- **1.2** — creative, occasionally nonsensical

```python
probs = np.log(probs + 1e-7) / temperature
probs = np.exp(probs) / np.sum(np.exp(probs))
predicted_idx = np.random.choice(len(probs), p=probs)
```

## Model Limitations

- **Vocabulary cap:** Words outside the top 5,000 are mapped to `<OOV>` — rare or technical words won't be predicted
- **Context window:** Only the last 15 tokens are used as context — longer inputs are truncated from the front
- **Domain-specific:** Trained on movie dialogue; performs best on conversational English. Academic, technical, or formal text may produce weaker predictions
- **No world knowledge:** The model learns statistical co-occurrence patterns, not meaning — grammatically plausible predictions are not guaranteed to be factually correct
- **Temperature sensitivity:** At temperature > 1.0, predictions can become incoherent; at temperature < 0.5, the model tends to repeat high-frequency words

## Project Structure

```
Next_Word_Predictor/
├── cornell_next_word.ipynb     # Full training pipeline
├── app.py                      # Streamlit inference app
├── next_word_lstm.h5           # Trained Keras model
├── tokenizer.pickle            # Fitted Keras Tokenizer
├── config.pickle               # MAX_LEN + VOCAB_SIZE config
├── training_curves.png         # Loss and accuracy plots
├── requirements.txt
└── README.md
```

## Setup / Run Locally

> Requires `next_word_lstm.h5`, `tokenizer.pickle`, and `config.pickle` — download from Kaggle output tab after training.

```bash
git clone https://github.com/<your-username>/Next_Word_Predictor.git
cd Next_Word_Predictor
pip install -r requirements.txt
streamlit run app.py
```

## Possible Improvements

- Use pretrained GloVe or Word2Vec embeddings instead of a trainable embedding layer — better generalisation with the same model size
- Upgrade to BiLSTM — reads sequences both forward and backward, typically 2–3% accuracy gain
- Train on the full Cornell corpus (100% instead of 20%) using a data generator to avoid RAM limits
- Add beam search decoding as an alternative to temperature sampling for more coherent multi-word predictions
- Experiment with a Transformer-based architecture (GPT-2 fine-tuning) as a v2 comparison
