"""
Causal Inference for Time Series
Methods for discovering and analyzing causal relationships in temporal data
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
import warnings


class GrangerCausality:
    """
    Granger Causality Testing
    Test if one time series can predict another
    """

    def __init__(self, max_lag: int = 10, significance_level: float = 0.05):
        """
        Args:
            max_lag: Maximum lag to test
            significance_level: Significance level for hypothesis testing
        """
        self.max_lag = max_lag
        self.significance_level = significance_level

    def test(
        self,
        cause: np.ndarray,
        effect: np.ndarray
    ) -> Dict[str, any]:
        """
        Test if 'cause' Granger-causes 'effect'

        Args:
            cause: Potential cause time series [N]
            effect: Potential effect time series [N]

        Returns:
            Dictionary with test results
        """
        # Prepare data [effect, cause]
        data = np.column_stack([effect, cause])

        results = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Run Granger causality test
            gc_results = grangercausalitytests(data, maxlag=self.max_lag, verbose=False)

            # Extract p-values for each lag
            p_values = []
            f_stats = []

            for lag in range(1, self.max_lag + 1):
                # Get F-test results
                f_test = gc_results[lag][0]['ssr_ftest']
                f_stats.append(f_test[0])  # F-statistic
                p_values.append(f_test[1])  # p-value

            results['p_values'] = p_values
            results['f_stats'] = f_stats
            results['min_p_value'] = min(p_values)
            results['best_lag'] = np.argmin(p_values) + 1
            results['is_causal'] = min(p_values) < self.significance_level

        return results

    def test_pairwise(
        self,
        data: np.ndarray,
        variable_names: Optional[List[str]] = None
    ) -> Dict[Tuple[int, int], Dict]:
        """
        Test pairwise Granger causality for all variables

        Args:
            data: Multivariate time series [N, num_variables]
            variable_names: Names of variables

        Returns:
            Dictionary of test results for each pair
        """
        num_vars = data.shape[1]

        if variable_names is None:
            variable_names = [f"Var_{i}" for i in range(num_vars)]

        results = {}

        for i in range(num_vars):
            for j in range(num_vars):
                if i == j:
                    continue

                # Test if i causes j
                result = self.test(data[:, i], data[:, j])
                results[(i, j)] = result
                results[(variable_names[i], variable_names[j])] = result

        return results

    def build_causal_graph(
        self,
        data: np.ndarray,
        variable_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Build causal graph adjacency matrix

        Args:
            data: Multivariate time series [N, num_variables]
            variable_names: Names of variables

        Returns:
            Adjacency matrix [num_vars, num_vars] where entry (i,j) = 1 if i causes j
        """
        num_vars = data.shape[1]
        adjacency_matrix = np.zeros((num_vars, num_vars))

        results = self.test_pairwise(data, variable_names)

        for i in range(num_vars):
            for j in range(num_vars):
                if i == j:
                    continue

                if results[(i, j)]['is_causal']:
                    adjacency_matrix[i, j] = 1

        return adjacency_matrix


class TransferEntropy:
    """
    Transfer Entropy for Causal Discovery
    Information-theoretic measure of causal influence
    """

    def __init__(self, bins: int = 10, lag: int = 1):
        """
        Args:
            bins: Number of bins for discretization
            lag: Time lag for transfer entropy
        """
        self.bins = bins
        self.lag = lag

    def _discretize(self, data: np.ndarray) -> np.ndarray:
        """Discretize continuous data into bins"""
        hist, bin_edges = np.histogram(data, bins=self.bins)
        return np.digitize(data, bin_edges[:-1])

    def _entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy"""
        _, counts = np.unique(data, return_counts=True)
        probabilities = counts / len(data)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def _joint_entropy(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate joint entropy"""
        joint = np.column_stack([data1, data2])
        unique_rows, counts = np.unique(joint, axis=0, return_counts=True)
        probabilities = counts / len(data1)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def _conditional_entropy(
        self,
        data: np.ndarray,
        condition: np.ndarray
    ) -> float:
        """Calculate conditional entropy H(data|condition)"""
        return self._joint_entropy(data, condition) - self._entropy(condition)

    def compute(
        self,
        source: np.ndarray,
        target: np.ndarray
    ) -> float:
        """
        Compute transfer entropy from source to target

        Args:
            source: Source time series [N]
            target: Target time series [N]

        Returns:
            Transfer entropy value
        """
        # Discretize data
        source_disc = self._discretize(source)
        target_disc = self._discretize(target)

        # Create lagged variables
        n = len(source) - self.lag

        target_present = target_disc[self.lag:]
        target_past = target_disc[:n]
        source_past = source_disc[:n]

        # TE = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})
        # = I(Y_t ; X_{t-1} | Y_{t-1})

        # H(Y_t | Y_{t-1})
        h_target_given_past = self._conditional_entropy(target_present, target_past)

        # H(Y_t | Y_{t-1}, X_{t-1})
        joint_condition = np.column_stack([target_past, source_past])
        h_target_given_both = self._joint_entropy(
            target_present,
            joint_condition
        ) - self._entropy(joint_condition.flatten())

        te = h_target_given_past - h_target_given_both

        return te


class InterventionAnalysis:
    """
    Intervention Analysis for Time Series
    Analyze the effect of interventions on time series
    """

    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model: Trained forecasting model
            device: Device to run on
        """
        self.model = model.to(device)
        self.device = device

    def intervene(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        intervention_variable: int,
        intervention_value: float,
        intervention_time: int
    ) -> torch.Tensor:
        """
        Perform intervention (do-operation)

        Args:
            x_enc, x_mark_enc, x_dec, x_mark_dec: Input data
            intervention_variable: Which variable to intervene on
            intervention_value: Value to set
            intervention_time: Time step to intervene

        Returns:
            Predictions after intervention
        """
        self.model.eval()

        # Clone inputs
        x_enc_intervened = x_enc.clone()
        x_dec_intervened = x_dec.clone()

        # Apply intervention
        if intervention_time < x_enc.size(1):
            x_enc_intervened[:, intervention_time, intervention_variable] = intervention_value
        else:
            dec_time = intervention_time - x_enc.size(1)
            x_dec_intervened[:, dec_time, intervention_variable] = intervention_value

        # Get prediction with intervention
        with torch.no_grad():
            prediction = self.model(
                x_enc_intervened,
                x_mark_enc,
                x_dec_intervened,
                x_mark_dec
            )

        return prediction

    def estimate_causal_effect(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        intervention_variable: int,
        intervention_values: List[float],
        intervention_time: int,
        target_variable: int
    ) -> Dict[str, np.ndarray]:
        """
        Estimate causal effect of intervention on target variable

        Args:
            x_enc, x_mark_enc, x_dec, x_mark_dec: Input data
            intervention_variable: Variable to intervene on
            intervention_values: List of values to try
            intervention_time: Time to intervene
            target_variable: Target variable to measure effect

        Returns:
            Dictionary with intervention values and corresponding effects
        """
        effects = []

        # Get baseline (no intervention)
        with torch.no_grad():
            baseline = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            baseline_value = baseline[:, :, target_variable].cpu().numpy()

        # Test each intervention value
        for value in intervention_values:
            prediction = self.intervene(
                x_enc, x_mark_enc, x_dec, x_mark_dec,
                intervention_variable, value, intervention_time
            )

            # Measure effect on target variable
            effect = prediction[:, :, target_variable].cpu().numpy() - baseline_value
            effects.append(effect.mean())

        return {
            'intervention_values': np.array(intervention_values),
            'effects': np.array(effects),
            'baseline': baseline_value.mean()
        }


class CounterfactualAnalysis:
    """
    Counterfactual Analysis for Time Series
    Answer "what if" questions about past events
    """

    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model: Trained forecasting model
            device: Device to run on
        """
        self.model = model.to(device)
        self.device = device

    def counterfactual_prediction(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        changes: Dict[Tuple[int, int], float]
    ) -> torch.Tensor:
        """
        Make counterfactual prediction with specified changes

        Args:
            x_enc, x_mark_enc, x_dec, x_mark_dec: Original input data
            changes: Dictionary mapping (time, variable) -> new_value

        Returns:
            Counterfactual predictions
        """
        self.model.eval()

        # Clone inputs
        x_enc_cf = x_enc.clone()
        x_dec_cf = x_dec.clone()

        # Apply changes
        for (time, variable), value in changes.items():
            if time < x_enc.size(1):
                x_enc_cf[:, time, variable] = value
            else:
                dec_time = time - x_enc.size(1)
                x_dec_cf[:, dec_time, variable] = value

        # Get counterfactual prediction
        with torch.no_grad():
            cf_prediction = self.model(x_enc_cf, x_mark_enc, x_dec_cf, x_mark_dec)

        return cf_prediction

    def analyze_counterfactual(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        y_actual: torch.Tensor,
        changes: Dict[Tuple[int, int], float]
    ) -> Dict[str, any]:
        """
        Analyze counterfactual scenario

        Args:
            x_enc, x_mark_enc, x_dec, x_mark_dec: Input data
            y_actual: Actual outcomes
            changes: Counterfactual changes

        Returns:
            Analysis results
        """
        # Get factual prediction
        with torch.no_grad():
            factual_pred = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # Get counterfactual prediction
        cf_pred = self.counterfactual_prediction(
            x_enc, x_mark_enc, x_dec, x_mark_dec, changes
        )

        # Calculate differences
        factual_error = torch.mean(torch.abs(factual_pred - y_actual))
        cf_error = torch.mean(torch.abs(cf_pred - y_actual))
        effect_size = torch.mean(torch.abs(cf_pred - factual_pred))

        return {
            'factual_prediction': factual_pred.cpu().numpy(),
            'counterfactual_prediction': cf_pred.cpu().numpy(),
            'actual': y_actual.cpu().numpy(),
            'factual_error': factual_error.item(),
            'counterfactual_error': cf_error.item(),
            'effect_size': effect_size.item()
        }


class CausalAttentionAnalyzer:
    """
    Analyze causal relationships using attention mechanisms
    """

    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model: Transformer model with attention
            device: Device to run on
        """
        self.model = model.to(device)
        self.device = device

    def extract_causal_weights(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor
    ) -> np.ndarray:
        """
        Extract attention weights as causal relationships

        Args:
            x_enc, x_mark_enc, x_dec, x_mark_dec: Input data

        Returns:
            Attention weights representing causal influences
        """
        self.model.eval()

        # This is a placeholder - actual implementation depends on model
        # You would need to modify the model to return attention weights

        with torch.no_grad():
            _ = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # Placeholder: return dummy attention weights
        # In practice, aggregate attention across layers and heads
        attention_weights = np.zeros((x_enc.size(1), x_enc.size(2)))

        return attention_weights

    def identify_causal_variables(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        threshold: float = 0.1
    ) -> List[int]:
        """
        Identify variables with strong causal influence

        Args:
            x_enc, x_mark_enc, x_dec, x_mark_dec: Input data
            threshold: Threshold for identifying important variables

        Returns:
            List of important variable indices
        """
        weights = self.extract_causal_weights(
            x_enc, x_mark_enc, x_dec, x_mark_dec
        )

        # Aggregate weights per variable
        variable_importance = np.mean(weights, axis=0)

        # Identify variables above threshold
        important_vars = np.where(variable_importance > threshold)[0].tolist()

        return important_vars


def compute_partial_correlation(
    data: np.ndarray,
    var1: int,
    var2: int,
    condition_vars: List[int]
) -> float:
    """
    Compute partial correlation between two variables given others

    Args:
        data: Multivariate time series [N, num_variables]
        var1, var2: Variables to compute correlation between
        condition_vars: Variables to condition on

    Returns:
        Partial correlation coefficient
    """
    from sklearn.linear_model import LinearRegression

    # Regress var1 on condition_vars
    reg1 = LinearRegression()
    reg1.fit(data[:, condition_vars], data[:, var1])
    residual1 = data[:, var1] - reg1.predict(data[:, condition_vars])

    # Regress var2 on condition_vars
    reg2 = LinearRegression()
    reg2.fit(data[:, condition_vars], data[:, var2])
    residual2 = data[:, var2] - reg2.predict(data[:, condition_vars])

    # Compute correlation of residuals
    partial_corr = np.corrcoef(residual1, residual2)[0, 1]

    return partial_corr


def detect_causal_lags(
    cause: np.ndarray,
    effect: np.ndarray,
    max_lag: int = 20
) -> Dict[str, any]:
    """
    Detect optimal causal lag between two time series

    Args:
        cause: Cause time series [N]
        effect: Effect time series [N]
        max_lag: Maximum lag to test

    Returns:
        Dictionary with optimal lag and correlation
    """
    correlations = []

    for lag in range(max_lag + 1):
        if lag == 0:
            corr = np.corrcoef(cause, effect)[0, 1]
        else:
            # Lag cause
            corr = np.corrcoef(cause[:-lag], effect[lag:])[0, 1]

        correlations.append(abs(corr))

    optimal_lag = np.argmax(correlations)
    max_correlation = correlations[optimal_lag]

    return {
        'optimal_lag': optimal_lag,
        'correlation': max_correlation,
        'all_correlations': correlations
    }
