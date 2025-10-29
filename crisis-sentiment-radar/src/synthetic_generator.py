"""
Synthetic Data Generator for Crisis Sentiment Analysis

This module provides utilities to generate realistic synthetic tweet data
for simulating various crisis scenarios and their impact on sentiment distribution.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any
import random


class CrashEventSimulator:
    """Simulates airline crash scenarios and their impact on sentiment."""
    
    def __init__(self, original_data: pd.DataFrame):
        """
        Initialize the crash event simulator.
        
        Args:
            original_data: Original tweet dataset for baseline patterns
        """
        self.original_data = original_data
        self.airlines = original_data['airline'].unique()
        
        # Crisis-specific vocabulary organized by sentiment
        self.crash_keywords = {
            'negative': [
                'crash', 'tragic', 'devastating', 'unsafe', 'dangerous', 'deadly', 
                'horrific', 'never flying again', 'scared', 'terrified', 
                'worried about safety', 'investigation', 'victims', 'prayers', 
                'heartbreaking', 'avoid this airline', 'safety concerns'
            ],
            'neutral': [
                'news report', 'investigation ongoing', 'authorities investigating', 
                'official statement', 'waiting for updates', 'facts unclear', 
                'monitoring situation', 'no comment'
            ],
            'positive': [
                'thoughts and prayers', 'supporting families', 'trust in safety measures',
                'rare occurrence', 'still confident', 'isolated incident'
            ]
        }
        
    def simulate_crash_timeline(self) -> List[Tuple[str, pd.DataFrame]]:
        """
        Generate complete crash scenario timeline.
        
        Returns:
            List of (period_name, dataframe) tuples for each time period
        """
        timeline_scenarios = []
        
        # Pre-crash: Normal distribution baseline
        pre_crash = self._generate_time_period(
            size=500,
            sentiment_probs=[0.63, 0.21, 0.16],
            period="Pre-Crash",
            affected_airline=None,
            hours_after=0
        )
        timeline_scenarios.append(('Pre-Crash (Normal)', pre_crash))
        
        # Hour 1-6: Immediate aftermath with extreme negative shift
        immediate = self._generate_time_period(
            size=800,
            sentiment_probs=[0.95, 0.04, 0.01],
            period="Immediate Aftermath",
            affected_airline="United",
            hours_after=3
        )
        timeline_scenarios.append(('Hours 1-6: Immediate Aftermath', immediate))
        
        # Day 1-3: Peak crisis with very negative sentiment
        peak_crisis = self._generate_time_period(
            size=1000,
            sentiment_probs=[0.88, 0.10, 0.02],
            period="Peak Crisis",
            affected_airline="United",
            hours_after=48
        )
        timeline_scenarios.append(('Days 1-3: Peak Crisis', peak_crisis))
        
        # Week 1: Ongoing concerns with gradual improvement
        week_1 = self._generate_time_period(
            size=600,
            sentiment_probs=[0.75, 0.18, 0.07],
            period="Week 1",
            affected_airline="United",
            hours_after=168
        )
        timeline_scenarios.append(('Week 1: Ongoing Concerns', week_1))
        
        # Month 1: Gradual recovery but lasting impact
        month_1 = self._generate_time_period(
            size=400,
            sentiment_probs=[0.70, 0.22, 0.08],
            period="Month 1",
            affected_airline="United",
            hours_after=720
        )
        timeline_scenarios.append(('Month 1: Gradual Recovery', month_1))
        
        return timeline_scenarios
    
    def _generate_time_period(self, size: int, sentiment_probs: List[float], 
                            period: str, affected_airline: str, 
                            hours_after: int) -> pd.DataFrame:
        """
        Generate tweets for a specific time period.
        
        Args:
            size: Number of tweets to generate
            sentiment_probs: [negative, neutral, positive] probabilities
            period: Period identifier
            affected_airline: Airline involved in crash
            hours_after: Hours since crash occurred
            
        Returns:
            DataFrame with synthetic tweet data
        """
        sentiments = np.random.choice(
            ['negative', 'neutral', 'positive'], 
            size=size, 
            p=sentiment_probs
        )
        
        synthetic_data = []
        for i in range(size):
            sentiment = sentiments[i]
            
            # Bias towards affected airline during crisis
            if affected_airline and period != "Pre-Crash" and np.random.random() < 0.6:
                airline = affected_airline
            else:
                airline = np.random.choice(self.airlines)
            
            text = self._generate_crash_context_text(
                sentiment, period, airline, affected_airline
            )
            
            synthetic_data.append({
                'text': text,
                'airline_sentiment': sentiment,
                'airline': airline,
                'period': period,
                'affected_airline': affected_airline,
                'hours_after_crash': hours_after,
                'is_affected_airline': airline == affected_airline,
                'synthetic': True,
                'timestamp': datetime.now() + timedelta(hours=hours_after + i/100)
            })
        
        return pd.DataFrame(synthetic_data)
    
    def _generate_crash_context_text(self, sentiment: str, period: str, 
                                   airline: str, affected_airline: str) -> str:
        """
        Generate contextually appropriate text based on crash timeline.
        
        Args:
            sentiment: Tweet sentiment (negative/neutral/positive)
            period: Time period after crash
            airline: Airline being discussed
            affected_airline: Airline involved in crash
            
        Returns:
            Generated tweet text
        """
        if period == "Pre-Crash":
            # Normal operational tweets
            normal_templates = [
                f"Flying with {airline} today",
                f"Good service from {airline}",
                f"Flight delayed with {airline}",
                f"Customer service issue with {airline}"
            ]
            return np.random.choice(normal_templates)
        
        # Crisis period tweets with appropriate vocabulary
        crash_words = self.crash_keywords[sentiment]
        
        if airline == affected_airline:
            # Tweets about the affected airline
            templates = self._get_affected_airline_templates(sentiment, airline, crash_words)
        else:
            # Tweets about other airlines (spillover effect)
            templates = self._get_other_airline_templates(
                sentiment, airline, affected_airline, crash_words
            )
        
        return np.random.choice(templates)
    
    def _get_affected_airline_templates(self, sentiment: str, airline: str, 
                                      crash_words: List[str]) -> List[str]:
        """Get tweet templates for the affected airline."""
        if sentiment == 'negative':
            return [
                f"{airline} crash is {np.random.choice(crash_words)}",
                f"Will never fly {airline} after this {np.random.choice(crash_words)} incident",
                f"{airline} safety record is {np.random.choice(crash_words)}",
                f"How can {airline} ensure this doesn't happen again? So {np.random.choice(crash_words)}",
                f"{airline} needs to address these {np.random.choice(crash_words)} safety issues"
            ]
        elif sentiment == 'neutral':
            return [
                f"{airline} {np.random.choice(crash_words)} - waiting for official statement",
                f"Following {airline} crash {np.random.choice(crash_words)}",
                f"{airline} incident under {np.random.choice(crash_words)}"
            ]
        else:  # positive
            return [
                f"{np.random.choice(crash_words)} for {airline} families",
                f"Supporting {airline} during this difficult time - {np.random.choice(crash_words)}",
                f"Still have faith in {airline} - {np.random.choice(crash_words)}"
            ]
    
    def _get_other_airline_templates(self, sentiment: str, airline: str, 
                                   affected_airline: str, crash_words: List[str]) -> List[str]:
        """Get tweet templates for other airlines (spillover effect)."""
        if sentiment == 'negative':
            return [
                f"After {affected_airline} crash, worried about flying {airline} too",
                f"All airlines including {airline} need better safety after {affected_airline} incident",
                f"Aviation safety concerns affect {airline} as well"
            ]
        elif sentiment == 'neutral':
            return [
                f"Flying {airline} - hope they have better safety than {affected_airline}",
                f"Checking {airline} safety record after {affected_airline} news"
            ]
        else:  # positive
            return [
                f"Trust {airline} safety more than {affected_airline}",
                f"{airline} has better safety record than {affected_airline}",
                f"Still confident in {airline} despite {affected_airline} incident"
            ]


class GeneralCrisisSimulator:
    """General purpose crisis simulator for various scenarios."""
    
    def __init__(self, original_data: pd.DataFrame):
        """Initialize with baseline data."""
        self.original_data = original_data
        self.entities = original_data['airline'].unique()
        
        self.sentiment_patterns = {
            'negative': ['terrible', 'worst', 'horrible', 'awful', 'disaster', 'failed'],
            'neutral': ['okay', 'fine', 'information', 'update', 'checking', 'question'],
            'positive': ['great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'best']
        }
    
    def generate_scenario(self, scenario_type: str, size: int = 1000) -> pd.DataFrame:
        """
        Generate data for different crisis scenarios.
        
        Args:
            scenario_type: Type of crisis ('financial', 'weather', 'positive_campaign')
            size: Number of synthetic records to generate
            
        Returns:
            DataFrame with synthetic data
        """
        if scenario_type == 'financial':
            sentiment_probs = [0.80, 0.15, 0.05]
        elif scenario_type == 'weather':
            sentiment_probs = [0.70, 0.25, 0.05]
        elif scenario_type == 'positive_campaign':
            sentiment_probs = [0.20, 0.30, 0.50]
        else:
            sentiment_probs = [0.63, 0.21, 0.16]  # Normal distribution
        
        sentiments = np.random.choice(
            ['negative', 'neutral', 'positive'], 
            size=size, 
            p=sentiment_probs
        )
        
        synthetic_data = []
        for i in range(size):
            sentiment = sentiments[i]
            entity = np.random.choice(self.entities)
            text = self._generate_general_text(sentiment, scenario_type, entity)
            
            synthetic_data.append({
                'text': text,
                'airline_sentiment': sentiment,
                'airline': entity,
                'scenario': scenario_type,
                'synthetic': True,
                'timestamp': datetime.now() + timedelta(hours=i/100)
            })
        
        return pd.DataFrame(synthetic_data)
    
    def _generate_general_text(self, sentiment: str, scenario: str, entity: str) -> str:
        """Generate text for general scenarios."""
        patterns = self.sentiment_patterns[sentiment]
        
        templates = [
            f"Experience with {entity} was {np.random.choice(patterns)}",
            f"Service from {entity} is {np.random.choice(patterns)}",
            f"Just used {entity} and it was {np.random.choice(patterns)}"
        ]
        
        return np.random.choice(templates)


# Example usage and testing functions
def test_crash_simulator():
    """Test the crash event simulator with sample data."""
    # Create sample data
    sample_data = pd.DataFrame({
        'airline': ['United', 'Delta', 'American', 'Southwest'] * 100,
        'airline_sentiment': np.random.choice(['negative', 'neutral', 'positive'], 400),
        'text': ['Sample tweet'] * 400
    })
    
    simulator = CrashEventSimulator(sample_data)
    timeline = simulator.simulate_crash_timeline()
    
    print("Generated timeline scenarios:")
    for period_name, data in timeline:
        print(f"{period_name}: {len(data)} tweets")
        sentiment_dist = data['airline_sentiment'].value_counts(normalize=True)
        print(f"  Sentiment: {dict(sentiment_dist)}")
    
    return timeline


if __name__ == "__main__":
    # Run test
    timeline = test_crash_simulator()
    print("\nCrash simulator test completed successfully!")