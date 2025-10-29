"""
Visualization Utilities for Sentiment Analysis and Distribution Drift

This module provides professional-grade visualization functions for
displaying sentiment analysis results and distribution drift patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class SentimentVisualizer:
    """Professional visualization utilities for sentiment analysis."""
    
    def __init__(self, style: str = 'default'):
        """
        Initialize visualizer with styling preferences.
        
        Args:
            style: matplotlib style to use ('default', 'seaborn', etc.)
        """
        plt.style.use(style)
        
        # Define consistent color schemes
        self.sentiment_colors = {
            'negative': '#d32f2f',    # Red
            'neutral': '#ff9800',     # Orange  
            'positive': '#4caf50'     # Green
        }
        
        self.sentiment_labels = {
            'negative': 'Negative',
            'neutral': 'Neutral',
            'positive': 'Positive'
        }
        
        # Severity level colors for drift visualization
        self.severity_colors = {
            'LOW': '#4caf50',
            'MODERATE': '#ff9800',
            'HIGH': '#f44336',
            'EXTREME': '#b71c1c'
        }
    
    def plot_crisis_timeline(self, timeline_data: List[Tuple[str, pd.DataFrame]], 
                           figsize: Tuple[int, int] = (25, 5)) -> None:
        """
        Create pie charts showing sentiment evolution across crisis timeline.
        
        Args:
            timeline_data: List of (period_name, dataframe) tuples
            figsize: Figure size tuple
        """
        n_periods = len(timeline_data)
        fig, axes = plt.subplots(1, n_periods, figsize=figsize)
        
        if n_periods == 1:
            axes = [axes]
        
        fig.suptitle('Crisis Impact: Sentiment Distribution Evolution', 
                     fontsize=18, fontweight='bold', y=1.02)
        
        for idx, (period_name, data) in enumerate(timeline_data):
            sentiment_counts = data['airline_sentiment'].value_counts()
            
            # Ensure consistent order and colors
            ordered_data = []
            ordered_labels = []
            ordered_colors = []
            
            for sentiment in ['negative', 'neutral', 'positive']:
                if sentiment in sentiment_counts.index:
                    ordered_data.append(sentiment_counts[sentiment])
                    ordered_labels.append(self.sentiment_labels[sentiment])
                    ordered_colors.append(self.sentiment_colors[sentiment])
            
            # Create pie chart
            wedges, texts, autotexts = axes[idx].pie(
                ordered_data, 
                labels=ordered_labels,
                colors=ordered_colors,
                autopct='%1.0f%%',
                startangle=90,
                textprops={'fontsize': 11, 'fontweight': 'bold'}
            )
            
            # Improve text readability
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
            
            # Clean period name for title
            clean_period = self._clean_period_name(period_name)
            axes[idx].set_title(clean_period, fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.show()
    
    def plot_sentiment_evolution(self, timeline_data: List[Tuple[str, pd.DataFrame]], 
                               figsize: Tuple[int, int] = (14, 8)) -> None:
        """
        Create line plot showing sentiment evolution over time.
        
        Args:
            timeline_data: List of (period_name, dataframe) tuples
            figsize: Figure size tuple
        """
        # Prepare data
        periods = []
        sentiment_props = {sentiment: [] for sentiment in ['negative', 'neutral', 'positive']}
        
        for period_name, data in timeline_data:
            periods.append(self._clean_period_name(period_name))
            sentiment_dist = data['airline_sentiment'].value_counts(normalize=True)
            
            for sentiment in ['negative', 'neutral', 'positive']:
                sentiment_props[sentiment].append(sentiment_dist.get(sentiment, 0))
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        for sentiment in ['negative', 'neutral', 'positive']:
            ax.plot(periods, sentiment_props[sentiment], 
                   'o-', linewidth=4, markersize=10, 
                   label=self.sentiment_labels[sentiment],
                   color=self.sentiment_colors[sentiment],
                   markerfacecolor='white', markeredgewidth=2)
        
        # Add annotations for negative sentiment
        for i, val in enumerate(sentiment_props['negative']):
            ax.annotate(f"{val:.0%}", 
                       xy=(i, val), 
                       xytext=(0, 15), textcoords='offset points',
                       ha='center', fontweight='bold', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor=self.sentiment_colors['negative'], 
                               alpha=0.7, edgecolor='none'),
                       color='white')
        
        # Styling
        ax.set_title('Sentiment Distribution Evolution During Crisis', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time Period After Crisis', fontweight='bold', fontsize=12)
        ax.set_ylabel('Proportion of Tweets', fontweight='bold', fontsize=12)
        ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1)
        
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def plot_airline_comparison(self, comparison_data: pd.DataFrame, 
                              figsize: Tuple[int, int] = (12, 7)) -> None:
        """
        Create bar chart comparing affected vs other airlines.
        
        Args:
            comparison_data: DataFrame with comparison metrics
            figsize: Figure size tuple
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(comparison_data))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, comparison_data['Affected_Airline'], 
                      width, label='Affected Airline', 
                      color='#b71c1c', alpha=0.8, edgecolor='white', linewidth=1)
        bars2 = ax.bar(x + width/2, comparison_data['Other_Airlines'], 
                      width, label='Other Airlines', 
                      color='#ef5350', alpha=0.8, edgecolor='white', linewidth=1)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.0%}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Styling
        ax.set_xlabel('Time Period After Crisis', fontweight='bold', fontsize=12)
        ax.set_ylabel('Proportion of Negative Tweets', fontweight='bold', fontsize=12)
        ax.set_title('Negative Sentiment: Affected Airline vs Others', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_data['Period'])
        ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
    
    def plot_drift_severity(self, drift_data: pd.DataFrame, 
                          figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot distribution drift severity over time.
        
        Args:
            drift_data: DataFrame with drift metrics
            figsize: Figure size tuple
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot total variation distance
        periods = [self._clean_period_name(p) for p in drift_data['Period']]
        ax.plot(range(len(periods)), drift_data['total_variation'], 
               'ro-', linewidth=3, markersize=10)
        
        # Add severity zones
        ax.axhspan(0.3, 1, alpha=0.2, color='red', label='EXTREME Drift')
        ax.axhspan(0.15, 0.3, alpha=0.2, color='orange', label='HIGH Drift')
        ax.axhspan(0.05, 0.15, alpha=0.2, color='yellow', label='MODERATE Drift')
        ax.axhspan(0, 0.05, alpha=0.2, color='green', label='LOW Drift')
        
        # Styling
        ax.set_title('Distribution Drift Severity Over Time', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Period', fontweight='bold')
        ax.set_ylabel('Total Variation Distance', fontweight='bold')
        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels(periods, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_sentiment_heatmap(self, data: pd.DataFrame, 
                             group_col: str, sentiment_col: str = 'airline_sentiment',
                             figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Create heatmap of sentiment distributions across groups.
        
        Args:
            data: DataFrame with sentiment data
            group_col: Column to group by (e.g., 'airline', 'topic')
            sentiment_col: Column with sentiment labels
            figsize: Figure size tuple
        """
        # Create cross-tabulation
        sentiment_crosstab = pd.crosstab(data[group_col], data[sentiment_col], 
                                       normalize='index')
        
        # Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(sentiment_crosstab, 
                   annot=True, 
                   cmap='RdYlBu_r',
                   fmt='.2f',
                   cbar_kws={'label': 'Proportion'})
        
        plt.title(f'Sentiment Distribution by {group_col.title()}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Sentiment')
        plt.ylabel(group_col.title())
        plt.tight_layout()
        plt.show()
    
    def plot_drift_dashboard(self, timeline_data: List[Tuple[str, pd.DataFrame]], 
                           drift_metrics: pd.DataFrame,
                           figsize: Tuple[int, int] = (20, 12)) -> None:
        """
        Create comprehensive dashboard with multiple visualizations.
        
        Args:
            timeline_data: Crisis timeline data
            drift_metrics: Drift detection results
            figsize: Figure size tuple
        """
        fig = plt.figure(figsize=figsize)
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[2, 2, 1])
        
        # 1. Sentiment evolution timeline
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_timeline_subplot(timeline_data, ax1)
        
        # 2. Drift severity
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_drift_severity_subplot(drift_metrics, ax2)
        
        # 3. Distribution comparison
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_distribution_comparison_subplot(timeline_data, ax3)
        
        # 4. Key metrics summary
        ax4 = fig.add_subplot(gs[:, 2])
        self._plot_metrics_summary_subplot(drift_metrics, ax4)
        
        plt.suptitle('Crisis Impact Dashboard: Distribution Drift Analysis', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def _clean_period_name(self, period_name: str) -> str:
        """Clean period names for display."""
        if 'Pre-Crash' in period_name:
            return 'Pre-Crash'
        elif 'Immediate' in period_name:
            return 'Hours 1-6'
        elif 'Peak' in period_name:
            return 'Days 1-3'
        elif 'Ongoing' in period_name:
            return 'Week 1'
        elif 'Gradual' in period_name:
            return 'Month 1'
        else:
            return period_name.split(':')[0] if ':' in period_name else period_name
    
    def _plot_timeline_subplot(self, timeline_data: List[Tuple[str, pd.DataFrame]], 
                             ax: plt.Axes) -> None:
        """Plot sentiment timeline in subplot."""
        periods = []
        neg_props = []
        
        for period_name, data in timeline_data:
            periods.append(self._clean_period_name(period_name))
            sentiment_dist = data['airline_sentiment'].value_counts(normalize=True)
            neg_props.append(sentiment_dist.get('negative', 0))
        
        ax.plot(periods, neg_props, 'ro-', linewidth=3, markersize=8)
        ax.set_title('Negative Sentiment Evolution')
        ax.set_ylabel('Proportion Negative')
        ax.grid(True, alpha=0.3)
    
    def _plot_drift_severity_subplot(self, drift_metrics: pd.DataFrame, 
                                   ax: plt.Axes) -> None:
        """Plot drift severity in subplot."""
        periods = [self._clean_period_name(p) for p in drift_metrics['Period']]
        ax.bar(periods, drift_metrics['total_variation'], 
               color=[self.severity_colors.get(s, 'gray') for s in drift_metrics['Drift_Severity']])
        ax.set_title('Drift Severity')
        ax.set_ylabel('TV Distance')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_distribution_comparison_subplot(self, timeline_data: List[Tuple[str, pd.DataFrame]], 
                                            ax: plt.Axes) -> None:
        """Plot distribution comparison in subplot."""
        baseline = timeline_data[0][1]['airline_sentiment'].value_counts(normalize=True)
        current = timeline_data[-1][1]['airline_sentiment'].value_counts(normalize=True)
        
        sentiments = ['negative', 'neutral', 'positive']
        baseline_vals = [baseline.get(s, 0) for s in sentiments]
        current_vals = [current.get(s, 0) for s in sentiments]
        
        x = np.arange(len(sentiments))
        width = 0.35
        
        ax.bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.7)
        ax.bar(x + width/2, current_vals, width, label='Current', alpha=0.7)
        
        ax.set_title('Baseline vs Current Distribution')
        ax.set_ylabel('Proportion')
        ax.set_xticks(x)
        ax.set_xticklabels([self.sentiment_labels[s] for s in sentiments])
        ax.legend()
    
    def _plot_metrics_summary_subplot(self, drift_metrics: pd.DataFrame, 
                                    ax: plt.Axes) -> None:
        """Plot key metrics summary in subplot."""
        ax.axis('off')
        
        # Calculate summary statistics
        max_drift = drift_metrics['total_variation'].max()
        avg_drift = drift_metrics['total_variation'].mean()
        extreme_periods = sum(drift_metrics['Drift_Severity'] == 'EXTREME')
        
        summary_text = f"""
KEY METRICS

Max Drift: {max_drift:.3f}
Avg Drift: {avg_drift:.3f}
Extreme Periods: {extreme_periods}

SEVERITY COUNTS:
{drift_metrics['Drift_Severity'].value_counts().to_string()}
        """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))


# Example usage and testing functions
def test_visualizer():
    """Test the visualizer with sample data."""
    # Create sample timeline data
    sample_timeline = [
        ('Pre-Crash (Normal)', pd.DataFrame({
            'airline_sentiment': np.random.choice(['negative', 'neutral', 'positive'], 
                                                500, p=[0.6, 0.2, 0.2])
        })),
        ('Hours 1-6: Immediate Aftermath', pd.DataFrame({
            'airline_sentiment': np.random.choice(['negative', 'neutral', 'positive'], 
                                                800, p=[0.95, 0.03, 0.02])
        }))
    ]
    
    # Test visualizer
    visualizer = SentimentVisualizer()
    print("Testing crisis timeline visualization...")
    visualizer.plot_crisis_timeline(sample_timeline)
    
    print("Visualizer test completed!")


if __name__ == "__main__":
    # Run test
    test_visualizer()