import cv2
import numpy as np
import piexif
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')

def preprocess_image(image_path):
    """Preprocess the image for model input."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image.astype('float32') / 255.0
    return image

def extract_metadata(image_path):
    """Extract metadata from the image."""
    exif_data = piexif.load(image_path)
    return exif_data

def preprocess_text(text):
    """Tokenize and prepare text data."""
    tokens = word_tokenize(text)
    return ' '.join(tokens)

def vectorize_text(texts):
    """Convert texts into numerical vectors."""
    vectorizer = TfidfVectorizer()
    text_vectors = vectorizer.fit_transform(texts)
    return text_vectors, vectorizer
