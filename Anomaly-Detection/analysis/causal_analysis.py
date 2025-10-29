from typing import Dict, Any
import pandas as pd

def granger_causality(df: pd.DataFrame, maxlag: int = 4) -> Dict[str, Any]:
    """Run pairwise Granger causality tests for all column pairs.
    Returns p-values for 'x causes y' tests across lags.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except Exception as e:
        raise RuntimeError("statsmodels is required for Granger causality.") from e

    cols = list(df.columns)
    results = {}
    for i, x in enumerate(cols):
        for j, y in enumerate(cols):
            if i == j:
                continue
            try:
                test = grangercausalitytests(df[[y, x]].dropna(), maxlag=maxlag, verbose=False)
                pv = {lag: test[lag][0]['ssr_ftest'][1] for lag in test.keys()}
                results[f"{x}→{y}"] = pv
            except Exception as e:
                results[f"{x}→{y}"] = {"error": str(e)}
    return results
