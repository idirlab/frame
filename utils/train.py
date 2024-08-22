from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Define Trainer parameters
def compute_metrics(p):
    print(p)
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    print(pred.sum(), pred.shape)
    print(labels.sum(), labels.shape)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, zero_division=0)
    precision = precision_score(y_true=labels, y_pred=pred, zero_division=0)
    f1 = f1_score(y_true=labels, y_pred=pred, zero_division=0)

    return {"accuracy": accuracy, "precision": precision, 
            "recall": recall, "f1": f1}