# model_training_ml.py

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM,
    SpatialDropout1D, Attention, Flatten, Dense, Dropout
)
import tensorflow as tf

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def train_evaluate_models(X_train, y_train, X_test, y_test, vocab_size, embedding_dim=128, max_len=150):
    # KNN hyperparameter tuning
    param_grid_knn = {'n_neighbors': [3, 5, 7, 9]}
    grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring='accuracy')
    grid_knn.fit(X_train, y_train)
    knn_clf = grid_knn.best_estimator_

    # Random Forest hyperparameter tuning
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, scoring='accuracy')
    grid_rf.fit(X_train, y_train)
    rf_clf = grid_rf.best_estimator_

    # Stacking classifier
    stacking_clf = StackingClassifier(
        estimators=[('rf', rf_clf), ('knn', knn_clf)],
        final_estimator=RandomForestClassifier(n_estimators=100)
    )
    stacking_clf.fit(X_train, y_train)

    # Hybrid CNN+BiLSTM+Attention cross-validation training
    y_train_array = np.array(y_train)
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    hybrid_scores = []

    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train_array[train_idx], y_train_array[val_idx]

        input_layer = Input(shape=(max_len,))
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
        model.fit(X_train_fold, y_train_fold, epochs=5, batch_size=32, verbose=1,
                  validation_data=(X_val_fold, y_val_fold))
        _, acc = model.evaluate(X_val_fold, y_val_fold, verbose=1)
        hybrid_scores.append(acc)

    print(f"\nCross-validation score (Hybrid CNN + BiLSTM + Attention): {np.mean(hybrid_scores):.4f}")

    # Cross-validation for RF and stacking classifier
    rf_cv_score = cross_val_score(rf_clf, X_train, y_train, cv=10).mean()
    stacking_cv_score = cross_val_score(stacking_clf, X_train, y_train, cv=10).mean()

    print("\nCross-validation Scores:")
    print(f'Random Forest: {rf_cv_score:.4f}')
    print(f'Stacking Classifier: {stacking_cv_score:.4f}')

    # Predictions on test set
    rf_pred = rf_clf.predict(X_test)
    stacking_pred = stacking_clf.predict(X_test)

    models_preds = {
        "Random Forest": rf_pred,
        "Stacking Classifier": stacking_pred,
    }

    for name, pred in models_preds.items():
        print(f"\nClassification Report for {name}:\n")
        print(classification_report(y_test, pred))
        plot_confusion_matrix(y_test, pred, name)

if __name__ == "__main__":
    import joblib

    # Load preprocessed data
    df = pd.read_csv('data/cleaned_data.csv')

    # Assuming text is tokenized & padded outside or use your preprocessing here:
    # Load tokenized padded sequences and label
    X = joblib.load('data/X_padded.pkl')  # You should save this during preprocessing
    y = df['sentiment'].values

    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # Vocabulary size and max_len should be consistent with tokenizer & padding
    vocab_size = 10000  # or load from tokenizer config
    max_len = 150
    embedding_dim = 128

    train_evaluate_models(X_train, y_train, X_test, y_test, vocab_size, embedding_dim, max_len)
