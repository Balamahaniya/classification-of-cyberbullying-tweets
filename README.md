# classification-of-cyberbullying-tweets


This repository focuses on the classification of cyberbullying tweets using a combination of classical machine learning models and advanced deep learning architectures with attention mechanisms. The goal is to accurately detect cyberbullying and categorize it into multiple classes.

---

## 🔍 Project Overview

We explore and compare multiple approaches:

### Machine Learning Models:
- **K-Nearest Neighbors (KNN)** with hyperparameter tuning
- **Random Forest (RF)** with GridSearch optimization
- **Stacking Classifier** combining RF and KNN with XGBoost as meta-learner

###  Deep Learning Models:
- **CNN + BiLSTM + Attention** with K-Fold cross-validation
- **CNN + BiGRU + Attention** architecture

Each model is evaluated using:
- Accuracy
- Classification Report
- Confusion Matrix
- Cross-Validation Score (for robustness)

---

## ⚙️ Installation
```bash
pip install -r requirements.txt

```markdown
## Dataset
The dataset consists of tweets labeled into 5 categories of cyberbullying.  
Preprocessing includes tokenization, cleaning, and padding.

## 🧪 Results Summary

| Model                      | Accuracy  |
|---------------------------|-----------|
| CNN + BiGRU + Attention   | 91.89%    |
| CNN + BiLSTM + Attention  | 91.84%    |
| Stacking Classifier       | 77.00%    |
| Random Forest             | 77.00%    |
