from sklearn.linear_model import LogisticRegression

def train_text_model(X_texts, y_texts):
    text_model = LogisticRegression()
    text_model.fit(X_texts, y_texts)
    return text_model
