# model_training.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Embedding, Conv1D, MaxPooling1D, LSTM, Bidirectional, Attention,
    Dense, Input, Dropout, SpatialDropout1D, Flatten
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def train_evaluate_model(df):
    # Ensure labels are integers
    df['sentiment'] = df['sentiment'].astype(int)

    # Tokenization
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(df['text_clean'])
    X_seq = tokenizer.texts_to_sequences(df['text_clean'])
    X_padded = pad_sequences(X_seq, maxlen=150)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, df['sentiment'], test_size=0.2, stratify=df['sentiment'], random_state=42
    )

    # Define Model
    embedding_dim = 128
    vocab_size = len(tokenizer.word_index) + 1

    input_layer = Input(shape=(150,))
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)
    conv = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(embedding)
    pool = MaxPooling1D(pool_size=2)(conv)
    bilstm = Bidirectional(LSTM(100, return_sequences=True))(pool)
    dropout = SpatialDropout1D(0.3)(bilstm)
    attention = Attention()([dropout, dropout])
    flat = Flatten()(attention)
    dense1 = Dense(64, activation='relu')(flat)
    dropout2 = Dropout(0.4)(dense1)
    output_layer = Dense(5, activation='softmax')(dropout2)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
        metrics=['accuracy']
    )

    # Early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train Model
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )

    # Predict
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Evaluation
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nFinal Accuracy: {accuracy:.4f}")

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(5), yticklabels=range(5))
    plt.title("Confusion Matrix - CNN + BiLSTM + Attention Model")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.show()

if __name__ == '__main__':
    # Load cleaned data here - adjust path if needed
    df = pd.read_csv('data/cleaned_data.csv')
    train_evaluate_model(df)
