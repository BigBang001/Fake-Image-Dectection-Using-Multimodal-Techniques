import os
import numpy as np
from preprocess import preprocess_image, extract_metadata, preprocess_text, vectorize_text
from image_model import train_image_model
from text_model import train_text_model
from metadata_model import train_metadata_model
from combine_models import combine_models
from evaluate import evaluate_model

# Load and preprocess data
image_paths = [os.path.join('data/images/real', fname) for fname in os.listdir('data/images/real')] + \
              [os.path.join('data/images/fake', fname) for fname in os.listdir('data/images/fake')]

X_images = np.array([preprocess_image(path) for path in image_paths])
y_images = np.array([1] * len(os.listdir('data/images/real')) + [0] * len(os.listdir('data/images/fake')))

metadata_paths = image_paths
X_metadata = np.array([extract_metadata(path) for path in metadata_paths])
y_metadata = y_images

with open('data/texts/captions.txt', 'r') as f:
    texts = f.readlines()
X_texts, vectorizer = vectorize_text([preprocess_text(text) for text in texts])
y_texts = y_images

# Train models
image_model = train_image_model(X_images, y_images)
text_model = train_text_model(X_texts, y_texts)
metadata_model = train_metadata_model(X_metadata, y_metadata)

# Combine models
combined_model = combine_models(image_model, text_model, metadata_model)

# Evaluate combined model (using a separate test set)
# Assume X_test and y_test are prepared test datasets
# evaluate_model(combined_model, X_test, y_test)
