from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB



def train_models(x_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=0),
        "Random Forest": RandomForestClassifier(),
        "MultinomialNB": MultinomialNB()
    }
    
    for name, model in models.items():
        model.fit(x_train, y_train)
        models[name] = model
    return models