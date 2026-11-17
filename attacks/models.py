"""Random Forest inference with predicted-domain semantic routing (Eq. 9)."""
import numpy as np
from collections import Counter

DOMAINS = ("medical", "financial", "legal", "code")
SLOTS = ("domain", "intent", "attribute", "keyword")


def require_training_coverage(original, retained, *, semantic=False, vocabulary=None):
    """Reject localization/filtering that erases any declared training target."""
    original, retained = list(original), list(retained)
    if not semantic:
        if set(original) != set(retained):
            raise ValueError("Successfully localized training data must cover every domain")
        return
    expected_domains = set(vocabulary) if vocabulary is not None else {row["domain"] for row in original}
    if {row["domain"] for row in retained} != expected_domains:
        raise ValueError("Successfully localized training data must cover every domain")
    for domain in expected_domains:
        for slot in SLOTS[1:]:
            expected = (set(vocabulary[domain][slot]) if vocabulary is not None else
                        {row[slot] for row in original if row["domain"] == domain})
            actual = {row[slot] for row in retained if row["domain"] == domain}
            if actual != expected:
                raise ValueError(f"Successfully localized training data must cover every {domain}/{slot} class; "
                                 f"missing={sorted(expected-actual)}, unknown={sorted(actual-expected)}")


def _matrix(X):
    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1] or not np.all(np.isfinite(values)):
        raise ValueError("X must be a nonempty finite two-dimensional feature matrix")
    return values


class AttributeAttack:
    channel = None
    semantic = False

    def __init__(self, seed=0, n_estimators=100, max_depth=10, n_jobs=1):
        self.seed, self.n_estimators = seed, n_estimators
        self.max_depth, self.n_jobs = max_depth, n_jobs
        self.classifier = None
        self.observation_options_ = None

    def _forest(self):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                      random_state=self.seed, n_jobs=self.n_jobs)

    def fit(self, X, y):
        X, y = _matrix(X), np.asarray(y, dtype=str)
        if y.ndim != 1 or len(y) != len(X) or not set(y) <= set(DOMAINS):
            raise ValueError("one valid domain label is required for every row")
        self.classifier = self._forest().fit(X, y)
        return self

    def predict(self, X):
        if self.classifier is None:
            raise RuntimeError("fit must be called before predict")
        return self.classifier.predict(_matrix(X))

    def predict_proba(self, X):
        if self.classifier is None:
            raise RuntimeError("fit must be called before predict_proba")
        return self.classifier.predict_proba(_matrix(X))

    def fit_observations(self, observations, labels, *, calibration=None,
                         window_ms=10.0, max_windows=512, spatial_bins=32, ablation="full"):
        """Train this variant directly from its channel's observations.

        Hardware localization failures are tracked separately from the successful
        auxiliary records used to fit the RF. The online prediction method retains
        every failed attempt as a None prediction.
        """
        observations, labels = list(observations), list(labels)
        if len(observations) != len(labels):
            raise ValueError("One label is required for each observation")
        self.observation_options_ = dict(calibration=calibration, window_ms=window_ms,
            max_windows=max_windows, spatial_bins=spatial_bins, ablation=ablation)
        from .hardware import LocalizationError
        features, targets = [], []
        self.training_failures_ = []
        for i, (observation, label) in enumerate(zip(observations, labels)):
            try:
                features.append(self._encode_observation(observation).flatten())
                targets.append(label)
            except LocalizationError:
                self.training_failures_.append(i)
        if not features:
            raise ValueError("No successfully acquired auxiliary observations")
        require_training_coverage(labels, targets, semantic=self.semantic)
        failed = set(self.training_failures_)
        self.training_indices_ = [i for i in range(len(labels)) if i not in failed]
        return self.fit(np.stack(features), targets)

    def _encode_observation(self, observation):
        from .ablation import encode_ablation
        from .hardware import LocalizationError
        if self.channel not in {"meta", "hw"} or self.observation_options_ is None:
            raise RuntimeError("Use a channel-specific class and fit_observations first")
        if any(observation.get(field, "completed") != "completed"
               for field in ("request_status", "acquisition_status")):
            raise LocalizationError("Request or observation acquisition failed")
        options = self.observation_options_
        kwargs = dict(window_ms=options["window_ms"], max_windows=options["max_windows"])
        if self.channel == "meta": kwargs["spatial_bins"] = options["spatial_bins"]
        return encode_ablation(observation, self.channel, options["ablation"],
                               options["calibration"], **kwargs)

    def predict_observations(self, observations):
        """Return one prediction per attempt, including None for failed localization."""
        from .hardware import LocalizationError
        result = []
        for observation in observations:
            try:
                features = self._encode_observation(observation).flatten()[None, :]
                result.append(self.predict(features)[0])
            except LocalizationError:
                result.append(None)
        return result


class SemanticAttack(AttributeAttack):
    """An independent domain RF plus three RFs per training domain.

    Keyword is a single label. Downstream predictions receive no true domain.
    This model is separate from the attribute classifier, including its domain RF.
    """
    semantic = True
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.domain_classifier = None
        self.heads = {}

    def fit(self, X, labels):
        X = _matrix(X)
        labels = list(labels)
        if len(labels) != len(X):
            raise ValueError("one semantic label record is required per feature row")
        for row in labels:
            if any(not isinstance(row.get(slot), str) or not row[slot] for slot in SLOTS):
                raise ValueError("domain, intent, attribute, keyword must be nonempty single labels")
            if row["domain"] not in DOMAINS:
                raise ValueError("unknown domain")
        domains = np.array([row["domain"] for row in labels])
        self.domain_classifier = self._forest().fit(X, domains)
        self.heads = {}
        for domain in sorted(set(domains)):
            mask = domains == domain
            self.heads[domain] = {}
            for slot in SLOTS[1:]:
                targets = np.array([row[slot] for row in labels])
                self.heads[domain][slot] = self._forest().fit(X[mask], targets[mask])
        # Count only examples actually passed to each fitted forest. Read its
        # classes_ as well, so a declared vocabulary cannot stand in for a model.
        domain_counts = Counter(str(domain) for domain in domains)
        if set(self.domain_classifier.classes_) != set(domain_counts):
            raise ValueError("Fitted domain classifier omitted a training class")
        self.training_support_ = dict(domain=dict(domain_counts), by_domain={})
        for domain, heads in self.heads.items():
            support = self.training_support_["by_domain"].setdefault(str(domain), {})
            for slot, forest in heads.items():
                counts = Counter(row[slot] for row in labels if row["domain"] == domain)
                if set(forest.classes_) != set(counts):
                    raise ValueError(f"Fitted {domain}/{slot} classifier omitted a training class")
                support[slot] = {str(label): counts[str(label)] for label in forest.classes_}
        return self

    def predict(self, X):
        if self.domain_classifier is None:
            raise RuntimeError("fit must be called before predict")
        X = _matrix(X)
        domains = self.domain_classifier.predict(X)
        output = [{"domain": str(domain)} for domain in domains]
        for domain in sorted(set(domains)):
            mask = domains == domain
            indices = np.flatnonzero(mask)
            for slot, classifier in self.heads[domain].items():
                predictions = classifier.predict(X[mask])
                for i, prediction in zip(indices, predictions):
                    output[i][slot] = str(prediction)
        return output

    def predict_proba(self, X):
        if self.domain_classifier is None:
            raise RuntimeError("fit must be called before predict_proba")
        return self.domain_classifier.predict_proba(_matrix(X))
