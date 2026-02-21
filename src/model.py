# src/model.py

from sklearn.linear_model import LogisticRegression


def build_model():
    """
    Build cost-sensitive logistic regression model
    False Negative cost = 10x
    """
    model = LogisticRegression(
        class_weight={0: 1, 1: 10},
        max_iter=1000
    )
    return model