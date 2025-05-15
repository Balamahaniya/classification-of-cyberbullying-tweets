# feature_extraction.py

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def tfidf_weighted_w2v_embedding(df, text_column='text_clean', label_column='sentiment'):
    # Tokenize text
    df.loc[:, 'tokenized_text'] = df[text_column].apply(lambda x: x.split())

    # Train Word2Vec model
    w2v_model = Word2Vec(sentences=df['tokenized_text'], vector_size=100, window=5, min_count=2, workers=4)

    # Compute TF-IDF scores
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df[text_column])
    tfidf_vocab = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))

    # Function to generate TF-IDF weighted Word2Vec vectors
    def tfidf_weighted_w2v(text):
        words = text.split()
        vector = np.zeros(100)
        weight_sum = 0
        for word in words:
            if word in w2v_model.wv:
                weight = tfidf_vocab.get(word, 1)
                vector += w2v_model.wv[word] * weight
                weight_sum += weight
        return vector / weight_sum if weight_sum != 0 else vector

    # Apply the embedding function
    df.loc[:, 'tfidf_w2v'] = df[text_column].apply(tfidf_weighted_w2v)

    # Convert list of vectors to NumPy array
    X = np.vstack(df['tfidf_w2v'])

    # Normalize feature vectors
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Prepare labels
    y = df[label_column]

    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # Print class distribution after SMOTE
    print("\nClass distribution after SMOTE:", np.bincount(y_train_balanced))

    return X_train_balanced, X_test, y_train_balanced, y_test

if __name__ == '__main__':
    # Example usage: Load cleaned data and run feature extraction

    df = pd.read_csv('data/cleaned_data.csv')  # Adjust path as needed

    X_train, X_test, y_train, y_test = tfidf_weighted_w2v_embedding(df,
                                text_column='cleaned_text', label_column='sentiment_encoded')

    print("Feature extraction and balancing done!")
