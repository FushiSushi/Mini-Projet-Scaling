from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

models = {
    "No Scaling": Pipeline([
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Standard": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "MinMax": Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Robust": Pipeline([
        ("scaler", RobustScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(name, ":", scores.mean())