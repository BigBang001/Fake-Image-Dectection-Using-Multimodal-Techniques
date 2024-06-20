from sklearn.ensemble import VotingClassifier

def combine_models(image_model, text_model, metadata_model):
    combined_model = VotingClassifier(estimators=[
        ('image', image_model),
        ('text', text_model),
        ('metadata', metadata_model)
    ], voting='soft')
    return combined_model
