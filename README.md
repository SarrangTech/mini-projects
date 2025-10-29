# 🚨 Airline Crash Sentiment Analysis: Distribution Drift Detection

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![BERTopic](https://img.shields.io/badge/BERTopic-Latest-green.svg)](https://github.com/MaartenGr/BERTopic)

## 🎯 Project Overview

This project demonstrates how external crisis events can cause **severe distribution drift** in real-time sentiment analysis systems. Using Twitter airline sentiment data and BERTopic, we analyze how an airline crash impacts sentiment distributions across the entire aviation industry.

### 🔍 Research Question
*"How does an airline crash affect sentiment distribution patterns, and how can we detect and measure this drift in real-time ML systems?"*

## 📊 Key Findings

| Phase | Timeline | Negative Sentiment | Drift Severity | Recovery Status |
|-------|----------|-------------------|----------------|-----------------|
| **Pre-Crisis** | Baseline | 60% | - | Normal |
| **Immediate** | Hours 1-6 | 96% ⬆️ | EXTREME | Critical |
| **Peak Crisis** | Days 1-3 | 88% | HIGH | Severe |
| **Ongoing** | Week 1 | 75% | MODERATE | Partial |
| **Recovery** | Month 1 | 71% | MODERATE | Incomplete |

### 💡 Critical Insights
- **⚡ Instant Impact**: Sentiment shifts from 60% to 96% negative within hours
- **🌊 Spillover Effect**: ALL airlines affected, not just the crashed airline
- **⏰ Long Recovery**: Even after 30 days, sentiment remains 11% above baseline
- **📈 Measurable Drift**: Statistical tests detect changes with p < 0.001

## 🛠️ Technical Stack

```python
# Core Libraries
pandas, numpy, matplotlib, seaborn, scikit-learn

# NLP & Topic Modeling
bertopic, sentence-transformers, transformers

# Statistical Analysis
scipy.stats (KS test, Wasserstein distance)

# Visualization
plotly, matplotlib with custom styling
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone or download the project
cd "D:\northeastern\mini projects\aircrash-sentiment-analysis"

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook airline_crash_sentiment_analysis.ipynb
```

### 2. Data Requirements
- **Dataset**: Twitter Airline Sentiment Dataset (14,640 tweets)
- **Path**: Place your `Tweets.csv.zip` in the data folder
- **Format**: CSV with columns: text, airline_sentiment, airline

### 3. Run Analysis
Execute all notebook cells sequentially to:
1. Load and preprocess data
2. Perform BERTopic analysis
3. Generate synthetic crash scenarios
4. Visualize distribution shifts
5. Calculate drift metrics

## 📁 Project Structure

```
📦 aircrash-sentiment-analysis/
├── 📊 airline_crash_sentiment_analysis.ipynb    # Main analysis notebook
├── 📋 requirements.txt                          # Dependencies
├── 📖 README.md                                # This file
├── 📄 analysis_summary.md                      # Detailed findings
├── 🗂️ data/
│   └── sample_tweets.csv                       # Sample data
├── 🖼️ visualizations/
│   ├── sentiment_evolution.png
│   ├── drift_severity.png
│   └── airline_comparison.png
├── 📜 src/
│   ├── synthetic_generator.py                  # Crisis simulation
│   ├── drift_detector.py                      # Statistical metrics
│   └── visualizer.py                          # Chart utilities
└── 📝 reports/
    └── linkedin_post.md                        # Social media content
```

## 🔬 Methodology

### Phase 1: Data Analysis
- **Topic Modeling**: BERTopic to discover 53 sentiment-driven topics
- **Preprocessing**: Text cleaning, noise removal, standardization
- **Baseline Establishment**: Pre-crisis sentiment distribution

### Phase 2: Crisis Simulation
- **Synthetic Data**: Realistic post-crash tweet generation
- **Timeline Modeling**: 5 phases from immediate to recovery
- **Spillover Effects**: Impact modeling on competitor airlines

### Phase 3: Drift Detection
- **Statistical Tests**: Kolmogorov-Smirnov, Wasserstein distance
- **Severity Classification**: EXTREME → HIGH → MODERATE → LOW
- **Metrics**: Total Variation Distance, probability shifts

## 📈 Key Visualizations

### 1. Sentiment Evolution Timeline
Professional line chart showing dramatic sentiment shifts across crisis phases

### 2. Distribution Pie Charts
Clean comparison of sentiment proportions from baseline through recovery

### 3. Airline Impact Analysis
Side-by-side comparison of affected vs. competitor airlines

### 4. Drift Severity Metrics
Statistical drift measurements with severity zone classifications

## 🎯 Real-World Applications

### ML Operations
- **Model Monitoring**: Detect when models need immediate retraining
- **Alert Systems**: Automated notifications for extreme distribution drift
- **Performance Tracking**: Quantify model degradation in real-time

### Business Intelligence
- **Crisis Management**: Rapid response to reputation threats
- **Risk Assessment**: Quantify spillover effects across business units
- **Competitive Analysis**: Monitor industry-wide sentiment patterns

### Research Applications
- **Social Media Analysis**: Study information propagation during crises
- **Behavioral Economics**: Analyze sentiment contagion effects
- **ML Robustness**: Test model resilience to external shocks

## 📊 Sample Output

```python
# Distribution Shift Example
PRE-CRISIS:     {"negative": 0.60, "neutral": 0.23, "positive": 0.17}
IMMEDIATE:      {"negative": 0.96, "neutral": 0.03, "positive": 0.01}
DRIFT_SEVERITY: "EXTREME (TV Distance: 0.352)"
KS_TEST:        "p < 0.001 (HIGHLY SIGNIFICANT)"
```

## 🔧 Advanced Features

### Drift Detection Algorithms
- **Total Variation Distance**: Measures overall distribution shift
- **Wasserstein Distance**: Earth mover's distance between distributions
- **Statistical Significance**: Multiple hypothesis tests for robust detection

### Synthetic Data Generation
- **Context-Aware**: Crisis-specific vocabulary and sentiment patterns
- **Temporal Modeling**: Realistic progression through crisis phases
- **Industry Effects**: Spillover impact on related businesses

## 📚 Academic Context

This project demonstrates critical concepts in:
- **Distribution Shift**: How real-world events affect ML model assumptions
- **Sentiment Analysis**: Advanced NLP techniques for social media data
- **Crisis Communication**: Information flow during emergency events
- **Statistical Detection**: Robust methods for identifying data drift

## 🎓 Learning Outcomes

After completing this project, you'll understand:
1. How to detect distribution drift in real-time systems
2. Advanced sentiment analysis with BERTopic
3. Statistical methods for measuring data shift
4. Crisis impact modeling and simulation
5. Professional data visualization techniques

## 📞 Contact & Collaboration

For questions, improvements, or research collaboration:
- **GitHub**: Create an issue or pull request
- **LinkedIn**: Share your analysis results
- **Academic**: Cite this work in research papers

## 🏆 Project Impact

This analysis has implications for:
- **Industry**: Airline reputation management systems
- **Academia**: Distribution drift research
- **Technology**: Real-time ML monitoring tools
- **Society**: Understanding crisis communication patterns

---

⭐ **Star this project if you find it useful for your research or work!** ⭐

*This project demonstrates the critical importance of monitoring distribution drift in production ML systems. Understanding how external events instantly impact model performance is essential for building robust AI applications.*