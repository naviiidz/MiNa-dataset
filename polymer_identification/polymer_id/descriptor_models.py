import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC

from .config import FEATURES_CSV, TABLE, fold_paths, parser, particle_id

warnings.filterwarnings("ignore")


def scaled(model):
    return make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0),
                         StandardScaler(), model)


MODELS = {
    "logistic_regression": (lambda **p: scaled(LogisticRegression(max_iter=3000, **p)), {"C": [0.01, 0.1, 1, 10]}),
    "knn": (lambda **p: scaled(KNeighborsClassifier(**p)), {"n_neighbors": [5, 15, 45], "weights": ["uniform", "distance"]}),
    "svm": (lambda **p: scaled(SVC(cache_size=2000, **p)), {"C": [0.3, 1, 3, 10]}),
    "random_forest": (lambda **p: RandomForestClassifier(n_estimators=300, n_jobs=4, random_state=0, **p),
                      {"max_depth": [6, 12, None], "min_samples_leaf": [1, 5, 20]}),
    "gradient_boosting": (lambda **p: HistGradientBoostingClassifier(random_state=0, **p),
                          {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31], "l2_regularization": [0.0, 1.0]}),
}


def load_features():
    feats = pd.read_csv(FEATURES_CSV)
    feats["particle_id"] = [particle_id(p, s, a) for p, s, a in zip(feats["polymer"], feats["source_image"], feats["annotation_id"])]
    return feats.set_index("particle_id")


def fit_and_predict(model_name, X, y):
    build, grid = MODELS[model_name]
    best = None
    for params in ParameterGrid(grid):
        model = build(**params).fit(X["train"], y["train"])
        score = f1_score(y["val"], model.predict(X["val"]), average="macro")
        if best is None or score > best[0]:
            best = (score, params, model)
    score, params, model = best
    return model_name, score, params, model.predict(X["test"])


def run_fold(fold, feats, features=TABLE, n_jobs=5):
    paths = fold_paths(fold)
    split = pd.read_csv(os.path.join(paths["splits"], "balanced.csv"))
    parts = {s: g for s, g in split.groupby("split")}
    X = {s: feats.loc[g["particle_id"], features].to_numpy(dtype=np.float64) for s, g in parts.items()}
    y = {s: g["label"].to_numpy() for s, g in parts.items()}
    out = Parallel(n_jobs=n_jobs)(delayed(fit_and_predict)(m, X, y) for m in MODELS)
    os.makedirs(paths["predictions"], exist_ok=True)
    for model_name, score, params, pred in out:
        pd.DataFrame({"particle_id": parts["test"]["particle_id"], "true": y["test"], "pred": pred}).to_csv(
            os.path.join(paths["predictions"], f"features_{model_name}.csv"), index=False)
        print(f"fold {fold} features + {model_name:20} val macro F1 {100 * score:5.1f}  {json.dumps(params)}")


def main():
    args = parser("Train classical classifiers on the measured particle descriptors "
                  "(diameter, area, aspect ratio, form factor, roundness, convexity).").parse_args()
    feats = load_features()
    for fold in args.folds:
        run_fold(fold, feats)


if __name__ == "__main__":
    main()
