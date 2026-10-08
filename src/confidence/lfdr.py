"""Statsmodels local-FDR selection with validation normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import local_fdr


def msp_to_zscore(msp: np.ndarray, null_logit_msp: np.ndarray) -> np.ndarray:
    score = np.log(np.clip(msp, 1e-6, 1 - 1e-6) / (1 - np.clip(msp, 1e-6, 1 - 1e-6)))
    mean, std = float(null_logit_msp.mean()), float(null_logit_msp.std(ddof=1))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Validation errors have zero MSP variance; LFDR cannot fit a null distribution")
    return (score - mean) / std


def annotate_msp_lfdr(calibration: pd.DataFrame, test: pd.DataFrame, q: float, score_column: str = "msp", correct_column: str = "correct") -> pd.DataFrame:
    if not 0 < q < 1:
        raise ValueError("q must be in (0, 1)")
    for frame in (calibration, test):
        if score_column not in frame:
            raise ValueError(f"Missing MSP column: {score_column}")
    if correct_column not in calibration:
        raise ValueError(f"Calibration lacks correctness column: {correct_column}")
    calibration_score = pd.to_numeric(calibration[score_column], errors="raise").to_numpy(float)
    wrong = ~calibration[correct_column].astype(str).str.lower().eq("true").to_numpy()
    if wrong.sum() < 2:
        raise ValueError("LFDR needs at least two wrong calibration samples")
    null_logit_msp = np.log(np.clip(calibration_score[wrong], 1e-6, 1 - 1e-6) / (1 - np.clip(calibration_score[wrong], 1e-6, 1 - 1e-6)))
    test_score = pd.to_numeric(test[score_column], errors="raise").to_numpy(float)
    z_test = msp_to_zscore(test_score, null_logit_msp)
    try:
        lfdr = local_fdr(z_test, null_proportion=float(wrong.mean()))
    except (ValueError, np.linalg.LinAlgError) as error:
        raise ValueError("Test MSP values cannot fit statsmodels local FDR") from error
    if not np.isfinite(lfdr).all():
        raise ValueError("Statsmodels local FDR returned non-finite values")
    output = test.copy()
    output["zscore"] = z_test
    output["lfdr"] = np.clip(lfdr, 0, 1)
    output["accept"] = output["lfdr"].le(q)
    return output
