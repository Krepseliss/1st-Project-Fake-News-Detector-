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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

def load_data(fake_csv, true_csv):
    fake = pd.read_csv('Fake.csv')
    true = pd.read_csv('True.csv')
    fake['class'] = 1  # Marking fake news as class 1
    true['class'] = 0  # Marking true news as class 0
    return pd.concat([fake, true], ignore_index=True)


def preprocess_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('\\W', " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', "", text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def preprocess_data(data):
    data['text'] = data['text'].apply(preprocess_text)
    return data

from models import train_models

def evaluate_models(models, x_test, y_test):
    for name, model in models.items():
        pred = model.predict(x_test)
        print(f"{name} Classification Report:\n{classification_report(y_test, pred)}\n")

def vectorize_data(x_train, x_test):
    vectorizer = TfidfVectorizer()
    xv_train = vectorizer.fit_transform(x_train)
    xv_test = vectorizer.transform(x_test)
    return vectorizer, xv_train, xv_test


def manual_test(news, models, vectorizer):
    processed_news = preprocess_text(news)
    vectorized_news = vectorizer.transform([processed_news])
    for name, model in models.items():
        prediction = model.predict(vectorized_news)[0]
        print(f"{name} Prediction: {'Possibly Hoax' if prediction == 1 else 'You can read it!'}")


def main():
    # Load and preprocess data
    data = load_data('Fake.csv', 'True.csv')
    data = preprocess_data(data)

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(data['text'], data['class'], test_size=0.25, random_state=42)

    # Vectorize text data and get the vectorizer
    vectorizer, xv_train, xv_test = vectorize_data(x_train, x_test)  # Capture the vectorizer here

    # Train models
    models = train_models(xv_train, y_train)

    # Evaluate models
    evaluate_models(models, xv_test, y_test)

    # Ask for user input for manual testing
    news = input("Paste your article: ")
    manual_test(news, models, vectorizer)  # Pass the vectorizer to manual_test

if __name__ == "__main__":
    main()