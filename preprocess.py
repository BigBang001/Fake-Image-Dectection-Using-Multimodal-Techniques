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
    return ' '.join(word_tokenize(text.lower()))

def vectorize_text(texts, vectorizer=None):
    """Convert texts into numerical vectors."""
    if vectorizer is None:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.8, min_df=5)
        text_vectors = vectorizer.fit_transform(texts)
    else:
        text_vectors = vectorizer.transform(texts)
    return text_vectors, vectorizer
