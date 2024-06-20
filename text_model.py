import nltk
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

nltk.download('punkt')

def preprocess_texts(texts):
    """Preprocess the texts by tokenizing."""
    return [' '.join(nltk.word_tokenize(text.lower())) for text in texts]

def train_text_model(X_texts, y_texts):
    """Train the text model using a Logistic Regression classifier."""
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_texts, y_texts, test_size=0.2, random_state=42)

    # Create a text processing and classification pipeline
    text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_df=0.8, min_df=5)),
        ('clf', LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0, class_weight='balanced'))
    ])
    
    # Train the model
    text_pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = text_pipeline.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")

    # Save the trained model for future use
    joblib.dump(text_pipeline, 'text_model.joblib')
    
    return text_pipeline

def load_text_model(model_path='text_model.joblib'):
    """Load a pre-trained text model."""
    return joblib.load(model_path)
