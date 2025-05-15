# data_preprocessing.py

import pandas as pd
import re
import string
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def main():
    # Load raw data - update the path and filename if needed
    df = pd.read_csv('data/raw_cyberbullying_tweets.csv')
    print(f"Loaded {len(df)} rows")

    # Clean the tweet text
    df['cleaned_text'] = df['tweet'].astype(str).apply(clean_text)

    # Encode sentiment labels (example: assuming 'label' column)
    le = LabelEncoder()
    df['sentiment_encoded'] = le.fit_transform(df['label'])

    # Save cleaned data
    df[['cleaned_text', 'sentiment_encoded']].to_csv('data/cleaned_data.csv', index=False)
    print("Cleaned data saved to data/cleaned_data.csv")

    # Optional: save label encoder classes for decoding later
    import joblib
    joblib.dump(le, 'models/label_encoder.pkl')
    print("Label encoder saved to models/label_encoder.pkl")

if __name__ == '__main__':
    main()
