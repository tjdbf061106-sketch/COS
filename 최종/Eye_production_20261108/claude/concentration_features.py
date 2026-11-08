"""Four concentration features with training-only median imputation.

Input percentages retain their original units (0..100). Molecular vectors are
never rescaled. Persist the fitted median alongside any deployed estimator.
"""

from __future__ import annotations

import numpy as np

NAMES = ["conc_log10_pct", "conc_pct", "conc_is_neat", "conc_undeclared"]


def _percentages(df):
    known = df.known_pct.to_numpy(dtype=np.float64)
    fallback = df.pct_sum_used.to_numpy(dtype=np.float64)
    undeclared = df.undeclared_pct.to_numpy(dtype=np.float64)
    for name, values in [
        ("known_pct", known),
        ("pct_sum_used", fallback),
        ("undeclared_pct", undeclared),
    ]:
        observed = values[~np.isnan(values)]
        if (
            not np.isfinite(observed).all()
            or (observed < 0).any()
            or (observed > 100).any()
        ):
            raise ValueError(
                f"{name} must be missing or a finite percentage in [0, 100]"
            )
    return np.where(np.isnan(known), fallback, known), undeclared


def fit_concentration_median(training_df):
    pct, _ = _percentages(training_df)
    observed = pct[~np.isnan(pct)]
    if not len(observed):
        raise ValueError(
            "Cannot fit concentration median: training percentages are all missing"
        )
    return float(np.median(observed))


def concentration_block(df, median):
    """Transform with a supplied training median; never fit on held-out rows."""
    if not np.isfinite(median) or not 0 <= median <= 100:
        raise ValueError("Training median must be a finite percentage in [0, 100]")
    pct, undeclared = _percentages(df)
    pct = np.where(np.isnan(pct), median, pct)
    undeclared = np.where(np.isnan(undeclared), 0.0, undeclared)
    block = np.column_stack(
        [
            np.log10(pct + 1e-3),
            pct / 100.0,
            (pct >= 90).astype(float),
            undeclared / 100.0,
        ]
    )
    if not np.isfinite(block).all():
        raise ValueError("Nonfinite concentration features")
    return block, list(NAMES), pct
