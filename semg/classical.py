"""Classical ML baselines (scikit-learn).

Every model is a Pipeline whose StandardScaler is fitted on the training data
only, so test subjects never leak into normalisation.
"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

from . import config as C


SGD_THRESHOLD = 100_000  # training windows above which "SVM" switches to SGD


def make_model(name, seed=C.SEED, n_jobs=-1, n_samples=None):
    if name == "SVM":
        # Linear SVM. Exact LinearSVC on small sets (within-subject); on big
        # LOSO folds (~140k windows) SGD with hinge loss gives the same
        # macro-F1 about 25x faster.
        name = "SVM-sgd" if (n_samples or 0) > SGD_THRESHOLD else "SVM-exact"
    if name == "LDA":
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif name == "kNN3":
        clf = KNeighborsClassifier(n_neighbors=3, n_jobs=n_jobs)
    elif name == "kNN5":
        clf = KNeighborsClassifier(n_neighbors=5, n_jobs=n_jobs)
    elif name == "SVM-sgd":
        clf = SGDClassifier(loss="hinge", alpha=1e-4, max_iter=50, tol=1e-3,
                            random_state=seed, n_jobs=n_jobs)
    elif name == "SVM-exact":
        clf = LinearSVC(dual=False, tol=1e-3, max_iter=500)
    elif name == "RF":
        return RandomForestClassifier(n_estimators=150, min_samples_leaf=2,
                                      n_jobs=n_jobs, random_state=seed)
    else:
        raise ValueError(f"Unknown classical model: {name}")
    return make_pipeline(StandardScaler(), clf)


def predict_scores(model, X):
    """Class scores for the simulator (probabilities when available)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return model.decision_function(X)
