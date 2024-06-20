from sklearn.ensemble import RandomForestClassifier

def train_metadata_model(X_metadata, y_metadata):
    metadata_model = RandomForestClassifier()
    metadata_model.fit(X_metadata, y_metadata)
    return metadata_model

