"""
Distribution Drift Detection Utilities

This module provides statistical methods for detecting and measuring
distribution drift in sentiment analysis systems.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from scipy.stats import wasserstein_distance, ks_2samp, chi2_contingency
import warnings


class DriftDetector:
    """Statistical methods for detecting distribution drift."""
    
    def __init__(self, baseline_data: pd.DataFrame, sentiment_column: str = 'airline_sentiment'):
        """
        Initialize drift detector with baseline data.
        
        Args:
            baseline_data: Reference dataset for comparison
            sentiment_column: Column containing sentiment labels
        """
        self.baseline_data = baseline_data
        self.sentiment_column = sentiment_column
        self.baseline_distribution = self._calculate_distribution(baseline_data)
        
    def detect_drift(self, target_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive drift detection analysis.
        
        Args:
            target_data: New data to compare against baseline
            
        Returns:
            Dictionary containing all drift metrics and significance tests
        """
        metrics = {}
        
        # Calculate basic distributions
        target_distribution = self._calculate_distribution(target_data)
        
        # Convert to numerical for distance calculations
        baseline_num = self._sentiment_to_numeric(self.baseline_data)
        target_num = self._sentiment_to_numeric(target_data)
        
        # Distance-based metrics
        metrics['wasserstein_distance'] = wasserstein_distance(baseline_num, target_num)
        metrics['total_variation_distance'] = self._total_variation_distance(
            self.baseline_distribution, target_distribution
        )
        
        # Statistical significance tests
        ks_stat, ks_pvalue = ks_2samp(baseline_num, target_num)
        metrics['ks_statistic'] = ks_stat
        metrics['ks_pvalue'] = ks_pvalue
        metrics['ks_significant'] = ks_pvalue < 0.05
        
        # Chi-square test for categorical distributions
        chi2_stat, chi2_pvalue = self._chi_square_test(
            self.baseline_data, target_data
        )
        metrics['chi2_statistic'] = chi2_stat
        metrics['chi2_pvalue'] = chi2_pvalue
        metrics['chi2_significant'] = chi2_pvalue < 0.05
        
        # Proportion differences
        for sentiment in ['negative', 'neutral', 'positive']:
            baseline_prop = self.baseline_distribution.get(sentiment, 0)
            target_prop = target_distribution.get(sentiment, 0)
            metrics[f'{sentiment}_diff'] = target_prop - baseline_prop
            metrics[f'{sentiment}_baseline'] = baseline_prop
            metrics[f'{sentiment}_target'] = target_prop
        
        # Drift severity classification
        metrics['drift_severity'] = self._classify_drift_severity(
            metrics['total_variation_distance']
        )
        
        # Overall drift score (0-1 scale)
        metrics['drift_score'] = min(1.0, metrics['total_variation_distance'] * 2)
        
        return metrics
    
    def _calculate_distribution(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate sentiment distribution proportions."""
        sentiment_counts = data[self.sentiment_column].value_counts(normalize=True)
        return sentiment_counts.to_dict()
    
    def _sentiment_to_numeric(self, data: pd.DataFrame) -> np.ndarray:
        """Convert sentiment labels to numeric values for distance calculations."""
        sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        return np.array([sentiment_map[s] for s in data[self.sentiment_column]])
    
    def _total_variation_distance(self, dist1: Dict[str, float], 
                                dist2: Dict[str, float]) -> float:
        """
        Calculate Total Variation Distance between two distributions.
        
        TV Distance = 0.5 * sum(|P(x) - Q(x)|) for all x
        """
        all_sentiments = set(dist1.keys()) | set(dist2.keys())
        tv_distance = 0.5 * sum(
            abs(dist1.get(sentiment, 0) - dist2.get(sentiment, 0))
            for sentiment in all_sentiments
        )
        return tv_distance
    
    def _chi_square_test(self, baseline: pd.DataFrame, 
                        target: pd.DataFrame) -> Tuple[float, float]:
        """Perform chi-square test for distribution independence."""
        # Create contingency table
        baseline_counts = baseline[self.sentiment_column].value_counts()
        target_counts = target[self.sentiment_column].value_counts()
        
        # Align indices
        all_sentiments = set(baseline_counts.index) | set(target_counts.index)
        
        baseline_aligned = [baseline_counts.get(s, 0) for s in all_sentiments]
        target_aligned = [target_counts.get(s, 0) for s in all_sentiments]
        
        contingency_table = np.array([baseline_aligned, target_aligned])
        
        # Perform chi-square test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
        
        return chi2_stat, p_value
    
    def _classify_drift_severity(self, tv_distance: float) -> str:
        """Classify drift severity based on Total Variation Distance."""
        if tv_distance > 0.3:
            return "EXTREME"
        elif tv_distance > 0.15:
            return "HIGH"
        elif tv_distance > 0.05:
            return "MODERATE"
        else:
            return "LOW"
    
    def batch_detect_drift(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Detect drift for multiple datasets at once.
        
        Args:
            datasets: Dictionary mapping dataset names to DataFrames
            
        Returns:
            DataFrame with drift metrics for each dataset
        """
        results = []
        
        for name, data in datasets.items():
            metrics = self.detect_drift(data)
            metrics['dataset_name'] = name
            metrics['dataset_size'] = len(data)
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def get_drift_summary(self, target_data: pd.DataFrame) -> str:
        """
        Generate human-readable drift summary.
        
        Args:
            target_data: Data to analyze for drift
            
        Returns:
            Formatted string summary of drift analysis
        """
        metrics = self.detect_drift(target_data)
        
        summary = f"""
DISTRIBUTION DRIFT ANALYSIS SUMMARY
{'='*50}

Dataset Size: {len(target_data)} samples
Drift Severity: {metrics['drift_severity']}
Drift Score: {metrics['drift_score']:.3f} (0-1 scale)

STATISTICAL TESTS:
- Total Variation Distance: {metrics['total_variation_distance']:.3f}
- Wasserstein Distance: {metrics['wasserstein_distance']:.3f}
- KS Test: p = {metrics['ks_pvalue']:.2e} ({'SIGNIFICANT' if metrics['ks_significant'] else 'NOT SIGNIFICANT'})
- Chi-square Test: p = {metrics['chi2_pvalue']:.2e} ({'SIGNIFICANT' if metrics['chi2_significant'] else 'NOT SIGNIFICANT'})

SENTIMENT CHANGES:
- Negative: {metrics['negative_baseline']:.1%} → {metrics['negative_target']:.1%} ({metrics['negative_diff']:+.1%})
- Neutral:  {metrics['neutral_baseline']:.1%} → {metrics['neutral_target']:.1%} ({metrics['neutral_diff']:+.1%})
- Positive: {metrics['positive_baseline']:.1%} → {metrics['positive_target']:.1%} ({metrics['positive_diff']:+.1%})

INTERPRETATION:
{self._get_interpretation(metrics)}
        """
        
        return summary.strip()
    
    def _get_interpretation(self, metrics: Dict[str, Any]) -> str:
        """Generate interpretation of drift results."""
        severity = metrics['drift_severity']
        
        if severity == "EXTREME":
            return "CRITICAL: Immediate model retraining required. Distribution has shifted dramatically."
        elif severity == "HIGH":
            return "WARNING: Significant drift detected. Consider model updates and increased monitoring."
        elif severity == "MODERATE":
            return "CAUTION: Moderate drift detected. Monitor closely and prepare for potential retraining."
        else:
            return "NORMAL: Low drift within acceptable bounds. Continue standard monitoring."


class TimeSeriesDriftDetector(DriftDetector):
    """Extended drift detector for time series analysis."""
    
    def __init__(self, baseline_data: pd.DataFrame, sentiment_column: str = 'airline_sentiment',
                 timestamp_column: str = 'timestamp'):
        """
        Initialize time series drift detector.
        
        Args:
            baseline_data: Reference dataset
            sentiment_column: Column with sentiment labels
            timestamp_column: Column with timestamps
        """
        super().__init__(baseline_data, sentiment_column)
        self.timestamp_column = timestamp_column
    
    def detect_drift_over_time(self, target_data: pd.DataFrame, 
                             window_size: str = '1H') -> pd.DataFrame:
        """
        Detect drift in rolling time windows.
        
        Args:
            target_data: Time series data to analyze
            window_size: Size of rolling window (e.g., '1H', '1D')
            
        Returns:
            DataFrame with drift metrics for each time window
        """
        # Ensure timestamp column is datetime
        if not pd.api.types.is_datetime64_any_dtype(target_data[self.timestamp_column]):
            target_data[self.timestamp_column] = pd.to_datetime(target_data[self.timestamp_column])
        
        # Set timestamp as index for resampling
        data_indexed = target_data.set_index(self.timestamp_column)
        
        # Group by time windows
        time_groups = data_indexed.groupby(pd.Grouper(freq=window_size))
        
        results = []
        for timestamp, group_data in time_groups:
            if len(group_data) > 0:  # Skip empty groups
                metrics = self.detect_drift(group_data.reset_index())
                metrics['timestamp'] = timestamp
                metrics['window_size'] = len(group_data)
                results.append(metrics)
        
        return pd.DataFrame(results)
    
    def plot_drift_timeline(self, drift_results: pd.DataFrame) -> None:
        """Plot drift metrics over time (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Distribution Drift Timeline', fontsize=16, fontweight='bold')
            
            # Total Variation Distance over time
            axes[0, 0].plot(drift_results['timestamp'], drift_results['total_variation_distance'])
            axes[0, 0].set_title('Total Variation Distance')
            axes[0, 0].set_ylabel('TV Distance')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Sentiment changes over time
            axes[0, 1].plot(drift_results['timestamp'], drift_results['negative_target'], 
                           label='Negative', color='red')
            axes[0, 1].plot(drift_results['timestamp'], drift_results['neutral_target'], 
                           label='Neutral', color='orange')
            axes[0, 1].plot(drift_results['timestamp'], drift_results['positive_target'], 
                           label='Positive', color='green')
            axes[0, 1].set_title('Sentiment Proportions')
            axes[0, 1].set_ylabel('Proportion')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # KS Test p-values
            axes[1, 0].semilogy(drift_results['timestamp'], drift_results['ks_pvalue'])
            axes[1, 0].axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
            axes[1, 0].set_title('KS Test p-values')
            axes[1, 0].set_ylabel('p-value (log scale)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Drift severity over time
            severity_map = {'LOW': 1, 'MODERATE': 2, 'HIGH': 3, 'EXTREME': 4}
            severity_numeric = [severity_map[s] for s in drift_results['drift_severity']]
            axes[1, 1].plot(drift_results['timestamp'], severity_numeric)
            axes[1, 1].set_title('Drift Severity Level')
            axes[1, 1].set_ylabel('Severity (1=Low, 4=Extreme)')
            axes[1, 1].set_ylim(0.5, 4.5)
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Install matplotlib for plotting functionality.")


# Example usage and testing functions
def test_drift_detector():
    """Test the drift detector with sample data."""
    # Create baseline data
    np.random.seed(42)
    baseline = pd.DataFrame({
        'airline_sentiment': np.random.choice(['negative', 'neutral', 'positive'], 
                                            1000, p=[0.6, 0.2, 0.2])
    })
    
    # Create shifted data
    shifted = pd.DataFrame({
        'airline_sentiment': np.random.choice(['negative', 'neutral', 'positive'], 
                                            1000, p=[0.9, 0.05, 0.05])
    })
    
    # Test drift detection
    detector = DriftDetector(baseline)
    metrics = detector.detect_drift(shifted)
    
    print("Drift Detection Test Results:")
    print(f"Drift Severity: {metrics['drift_severity']}")
    print(f"TV Distance: {metrics['total_variation_distance']:.3f}")
    print(f"KS Test p-value: {metrics['ks_pvalue']:.2e}")
    
    # Test summary
    summary = detector.get_drift_summary(shifted)
    print("\nDrift Summary:")
    print(summary)
    
    return metrics


if __name__ == "__main__":
    # Run test
    test_results = test_drift_detector()
    print("\nDrift detector test completed successfully!")