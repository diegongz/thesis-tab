from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import skorch
import os

def get_default_scores(y_test, y_pred):

    prediction = np.argmax(y_pred, axis=1)

    balanced_accuracy = balanced_accuracy_score(y_test, prediction)
    accuracy = accuracy_score(y_test, prediction)
    f1 = f1_score(y_test, prediction, average='weighted')
    precision = precision_score(y_test, prediction, average='weighted')
    recall = recall_score(y_test, prediction, average='weighted')

    metrics = {
        'balanced_accuracy': balanced_accuracy,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

    return metrics


def list_models(dir):
    model_path = []
    for f in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, f)):
            model_path.append(f)

    return model_path

def load_model(model, checkpoint_dir):
    checkpoint = skorch.callbacks.Checkpoint(dirname=checkpoint_dir)
    model.initialize()
    model.load_params(checkpoint=checkpoint)
    return model
