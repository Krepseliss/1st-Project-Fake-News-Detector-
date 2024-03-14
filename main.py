import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

def load_data(fake_csv, true_csv):
    fake = pd.read_csv('Fake.csv')
    true = pd.read_csv('True.csv')
    fake['class'] = 1  # Marking fake news as class 1
    true['class'] = 0  # Marking true news as class 0
    return pd.concat([fake, true], ignore_index=True)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        preprocessed_texts = []
        for text in X:
            text = text.lower()
            text = re.sub('\[.*?\]', '', text)
            text = re.sub('\\W', " ", text)
            text = re.sub('https?://\S+|www\.\S+', '', text)
            text = re.sub('<.*?>+', "", text)
            text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
            text = re.sub('\n', '', text)
            text = re.sub('\w*\d\w*', '', text)
            preprocessed_texts.append(text)
        return preprocessed_texts

pipeline = Pipeline([
    ('preprocessor', TextPreprocessor()),
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

def manual_test(news, pipeline):
    processed_news = [news]
    prediction = pipeline.predict(processed_news)[0]
    print(f"Prediction: {'Possibly Hoax' if prediction == 1 else 'You can read it!'}")



def main():
    # Load and preprocess data
    data = load_data('Fake.csv', 'True.csv')

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(data['text'], data['class'], test_size=0.25, random_state=42)

    # Train model
    models = pipeline.fit(x_train, y_train)

    # Evaluate model
    predictions = pipeline.predict(x_test)
    print(classification_report(y_test, predictions))

    # Ask for user input for manual testing
    news = input("Paste your article: ")
    manual_test(news, pipeline)

if __name__ == "__main__":
    main()

