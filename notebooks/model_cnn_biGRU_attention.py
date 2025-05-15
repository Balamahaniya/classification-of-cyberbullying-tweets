# model_cnn_biGRU_attention.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Embedding, Conv1D, MaxPooling1D, GRU, Bidirectional, Dense, Input,
    Dropout, SpatialDropout1D, Flatten, Attention
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

def run_model(df):
    # Tokenization
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(df['text_clean'])
    X_seq = tokenizer.texts_to_sequences(df['text_clean'])
    X_padded = pad_sequences(X_seq, maxlen=150)

    # Labels
    y = df['sentiment'].astype(int)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y, test_size=0.2, stratify=y, random_state=42
    )

    # Model Architecture
    embedding_dim = 128
    vocab_size = len(tokenizer.word_index) + 1

    input_layer = Input(shape=(150,))
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)

    conv1 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(embedding_layer)
    max_pool = MaxPooling1D(pool_size=2)(conv1)

    bigru = Bidirectional(GRU(100, return_sequences=True))(max_pool)
    spatial_dropout = SpatialDropout1D(0.3)(bigru)

    attention_output = Attention()([spatial_dropout, spatial_dropout])

    flatten = Flatten()(attention_output)
    dense1 = Dense(64, activation='relu')(flatten)
    dropout = Dropout(0.4)(dense1)

    output_layer = Dense(5, activation='softmax')(dropout)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
        metrics=['accuracy']
    )

    # Train Model
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test)
    )

    # Predictions & Evaluation
    y_pred = np.argmax(model.predict(X_test), axis=1)

    print("\nClassification Report for CNN + BiGRU + Attention Model:\n", classification_report(y_test, y_pred))
    print("\nAccuracy:", accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - CNN + BiGRU + Attention")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Save model and tokenizer
    model.save('models/cnn_bigru_attention_model.h5')
    with open('models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print("Model and tokenizer saved in 'models/' directory.")

if __name__ == "__main__":
    import pandas as pd
    # Load your cleaned data CSV here (update path if needed)
    df = pd.read_csv('data/cleaned_data.csv')
    run_model(df)
