from sklearn.ensemble import VotingClassifier
from text_model import load_text_model
import joblib

def combine_models(image_model, text_model_path, metadata_model):
    """Combine the individual models into a voting classifier."""
    text_model = load_text_model(text_model_path)
    combined_model = VotingClassifier(estimators=[
        ('image', image_model),
        ('text', text_model),
        ('metadata', metadata_model)
    ], voting='soft')
    return combined_model
