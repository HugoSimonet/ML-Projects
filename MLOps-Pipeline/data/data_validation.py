"""Data validation module for ensuring data quality."""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)
        self.passed = False

    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)


class DataValidator:
    """Validates data quality and schema."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize data validator.

        Args:
            config: Configuration dictionary containing:
                - schema: Expected data schema
                - quality_checks: List of quality checks to perform
                - thresholds: Thresholds for various checks
        """
        self.config = config
        self.schema = config.get('schema', {})
        self.quality_checks = config.get('quality_checks', [])
        self.thresholds = config.get('thresholds', {})

        logger.info("Initialized DataValidator")

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate the data.

        Args:
            data: DataFrame to validate

        Returns:
            ValidationResult with details
        """
        logger.info(f"Validating data with {len(data)} rows")

        result = ValidationResult(passed=True)

        # Schema validation
        self._validate_schema(data, result)

        # Quality checks
        self._validate_completeness(data, result)
        self._validate_consistency(data, result)
        self._validate_distributions(data, result)
        self._validate_duplicates(data, result)

        # Custom checks
        for check in self.quality_checks:
            self._run_custom_check(data, check, result)

        logger.info(f"Validation completed: passed={result.passed}, "
                   f"errors={len(result.errors)}, warnings={len(result.warnings)}")

        return result

    def _validate_schema(self, data: pd.DataFrame, result: ValidationResult):
        """Validate data schema."""
        if not self.schema:
            return

        # Check required columns
        required_columns = self.schema.get('required_columns', [])
        missing_columns = set(required_columns) - set(data.columns)

        if missing_columns:
            result.add_error(f"Missing required columns: {missing_columns}")

        # Check data types
        expected_dtypes = self.schema.get('dtypes', {})
        for col, expected_dtype in expected_dtypes.items():
            if col in data.columns:
                actual_dtype = str(data[col].dtype)
                if not self._is_compatible_dtype(actual_dtype, expected_dtype):
                    result.add_error(
                        f"Column '{col}' has dtype {actual_dtype}, "
                        f"expected {expected_dtype}"
                    )

    def _is_compatible_dtype(self, actual: str, expected: str) -> bool:
        """Check if data types are compatible."""
        # Simplified dtype compatibility check
        type_groups = {
            'numeric': ['int', 'float', 'int64', 'float64', 'int32', 'float32', 'int16', 'float16'],
            'string': ['object', 'string'],
            'datetime': ['datetime64']
        }

        # Check if expected is a general type (like 'numeric')
        if expected in type_groups:
            return any(t in actual.lower() for t in type_groups[expected])

        # Check if types are in the same group
        for group, types in type_groups.items():
            if expected in types and any(t in actual.lower() for t in types):
                return True

        return actual == expected

    def _validate_completeness(self, data: pd.DataFrame, result: ValidationResult):
        """Validate data completeness (missing values)."""
        max_missing_pct = self.thresholds.get('max_missing_pct', 0.1)

        for col in data.columns:
            missing_pct = data[col].isnull().sum() / len(data)

            if missing_pct > max_missing_pct:
                result.add_error(
                    f"Column '{col}' has {missing_pct:.2%} missing values "
                    f"(threshold: {max_missing_pct:.2%})"
                )
            elif missing_pct > 0:
                result.add_warning(
                    f"Column '{col}' has {missing_pct:.2%} missing values"
                )

        result.metrics['completeness'] = 1 - data.isnull().sum().sum() / data.size

    def _validate_consistency(self, data: pd.DataFrame, result: ValidationResult):
        """Validate data consistency."""
        # Check for infinite values in numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if np.isinf(data[col]).any():
                result.add_error(f"Column '{col}' contains infinite values")

        # Check for outliers using IQR method
        outlier_threshold = self.thresholds.get('outlier_threshold', 3.0)

        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1

            outliers = ((data[col] < Q1 - outlier_threshold * IQR) |
                       (data[col] > Q3 + outlier_threshold * IQR))

            outlier_pct = outliers.sum() / len(data)

            if outlier_pct > 0.05:  # More than 5% outliers
                result.add_warning(
                    f"Column '{col}' has {outlier_pct:.2%} outliers"
                )

    def _validate_distributions(self, data: pd.DataFrame, result: ValidationResult):
        """Validate data distributions."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        distributions = {}
        for col in numeric_cols:
            distributions[col] = {
                'mean': float(data[col].mean()),
                'std': float(data[col].std()),
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'skewness': float(data[col].skew())
            }

        result.metrics['distributions'] = distributions

    def _validate_duplicates(self, data: pd.DataFrame, result: ValidationResult):
        """Validate for duplicate records."""
        duplicates = data.duplicated().sum()
        duplicate_pct = duplicates / len(data)

        max_duplicate_pct = self.thresholds.get('max_duplicate_pct', 0.01)

        if duplicate_pct > max_duplicate_pct:
            result.add_error(
                f"Dataset has {duplicate_pct:.2%} duplicate records "
                f"(threshold: {max_duplicate_pct:.2%})"
            )
        elif duplicates > 0:
            result.add_warning(f"Dataset has {duplicates} duplicate records")

        result.metrics['duplicate_pct'] = duplicate_pct

    def _run_custom_check(self, data: pd.DataFrame, check: Dict[str, Any],
                         result: ValidationResult):
        """Run a custom validation check."""
        check_type = check.get('type')

        if check_type == 'range':
            self._check_range(data, check, result)
        elif check_type == 'unique':
            self._check_unique(data, check, result)
        elif check_type == 'pattern':
            self._check_pattern(data, check, result)

    def _check_range(self, data: pd.DataFrame, check: Dict[str, Any],
                    result: ValidationResult):
        """Check if values are within expected range."""
        column = check['column']
        min_val = check.get('min')
        max_val = check.get('max')

        if column not in data.columns:
            result.add_error(f"Column '{column}' not found for range check")
            return

        if min_val is not None and (data[column] < min_val).any():
            result.add_error(
                f"Column '{column}' has values below minimum {min_val}"
            )

        if max_val is not None and (data[column] > max_val).any():
            result.add_error(
                f"Column '{column}' has values above maximum {max_val}"
            )

    def _check_unique(self, data: pd.DataFrame, check: Dict[str, Any],
                     result: ValidationResult):
        """Check if column values are unique."""
        column = check['column']

        if column not in data.columns:
            result.add_error(f"Column '{column}' not found for unique check")
            return

        if data[column].duplicated().any():
            result.add_error(f"Column '{column}' contains duplicate values")

    def _check_pattern(self, data: pd.DataFrame, check: Dict[str, Any],
                      result: ValidationResult):
        """Check if string values match expected pattern."""
        column = check['column']
        pattern = check.get('pattern')

        if column not in data.columns:
            result.add_error(f"Column '{column}' not found for pattern check")
            return

        if pattern:
            invalid = ~data[column].astype(str).str.match(pattern)
            if invalid.any():
                result.add_warning(
                    f"Column '{column}' has {invalid.sum()} values "
                    f"not matching pattern '{pattern}'"
                )
