import os
import numpy as np
from preprocess import preprocess_image, extract_metadata, preprocess_text, vectorize_text
from image_model import train_image_model
from text_model import train_text_model, load_text_model
from metadata_model import train_metadata_model
from combine_models import combine_models
from evaluate import evaluate_model
from sklearn.model_selection import train_test_split
import joblib

def load_data():
    """Load and preprocess image, text, and metadata data."""
    image_paths = [os.path.join('data/images/real', fname) for fname in os.listdir('data/images/real')] + \
                  [os.path.join('data/images/fake', fname) for fname in os.listdir('data/images/fake')]

    X_images = np.array([preprocess_image(path) for path in image_paths])
    y_images = np.array([1] * len(os.listdir('data/images/real')) + [0] * len(os.listdir('data/images/fake')))

    metadata_paths = image_paths
    X_metadata = np.array([list(extract_metadata(path).values()) for path in metadata_paths])
    y_metadata = y_images

    with open('data/texts/captions.txt', 'r') as f:
        texts = f.readlines()
    texts = [preprocess_text(text) for text in texts]
    
    X_texts_train, X_texts_test, y_texts_train, y_texts_test = train_test_split(texts, y_images, test_size=0.2, random_state=42)
    X_texts_train_vec, vectorizer = vectorize_text(X_texts_train)
    X_texts_test_vec, _ = vectorize_text(X_texts_test, vectorizer)
    
    return X_images, y_images, X_texts_train_vec, y_texts_train, X_texts_test_vec, y_texts_test, X_metadata, y_metadata, vectorizer

if __name__ == "__main__":
    # Load and preprocess data
    X_images, y_images, X_texts_train_vec, y_texts_train, X_texts_test_vec, y_texts_test, X_metadata, y_metadata, vectorizer = load_data()

    # Train models
    image_model = train_image_model(X_images, y_images)
    text_model = train_text_model(X_texts_train_vec, y_texts_train)
    metadata_model = train_metadata_model(X_metadata, y_metadata)

    # Combine models
    combined_model = combine_models(image_model, text_model, metadata_model)

    # Evaluate text model
    evaluate_model(text_model, X_texts_test_vec, y_texts_test)

    # Save vectorizer for future use
    joblib.dump(vectorizer, 'vectorizer.joblib')

    # Combine models and evaluate (assuming X_test and y_test are prepared test datasets)
    # evaluate_model(combined_model, X_test, y_test)
