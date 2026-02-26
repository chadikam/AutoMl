"""
Feature Flags for AutoML MVP
==============================

Controls which features are enabled in production.
Disabled features are NOT deleted — their code remains intact for future reactivation.

Usage:
    from app.feature_flags import FeatureFlags
    if FeatureFlags.ENABLE_UNSUPERVISED:
        ...  # unsupervised path
    else:
        raise HTTPException(...)  # reject gracefully

TODO: Re-enable in v2 after full validation
"""


class FeatureFlags:
    """
    Centralized feature toggle for MVP launch.

    Set a flag to True to re-enable its feature.
    All guards in the codebase reference these constants so a single
    change here propagates everywhere.
    """

    # ── Unsupervised Learning ────────────────────────────────────────
    # Covers: clustering, dimensionality reduction, anomaly detection
    # TODO: Re-enable in v2 after full validation
    ENABLE_UNSUPERVISED: bool = False

    # ── TF-IDF / Text Processing ─────────────────────────────────────
    # Covers: TfidfVectorizer-based text column handling in adaptive preprocessing
    # TODO: Re-enable in v2 after full validation
    ENABLE_TEXT_PROCESSING: bool = False
