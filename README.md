# BeatScope: Spotify Analytics

**Personal Project** - A personal data-driven platform transforming 10+ years of Spotify streaming history into actionable insights through advanced analytics and interactive visualizations.

## Overview

This is a **personal data-driven platform** that analyzes my personal Spotify streaming history to reveal listening patterns, music preferences, and behavioral insights across 10+ years of data. Built a scalable Python data processing pipeline to parse large JSON datasets (100K+ records) and implemented temporal pattern analysis for behavioral modeling. Works entirely with exported Spotify data files.

**Core Components:**
- **Data Processing**: Parse Spotify JSON exports from raw data folder
- **Analytics Engine**: Extract temporal patterns, listening behavior, and music preferences  
- **Interactive Dashboard**: Streamlit-based visualization with filtering and export capabilities

## Personal Data Privacy

⚠️ **Important**: This project uses my personal Spotify data. All data processing happens locally and no personal information is shared.

**Data Sources:**
- **Spotify Account Data**: Downloaded from Spotify's privacy settings
- **Extended Streaming History**: Personal listening history from Spotify's data export feature

## Features

### Data Processing
- **Multi-Format Parser**: Process music, podcast, audiobook, and video streaming data
- **Raw Data Analysis**: Works entirely with exported JSON files
- **Data Enrichment**: Create derived metrics like skip rates, session patterns, and listening efficiency
- **Scalable Pipeline**: Handle large datasets (100K+ records) with memory optimization

### Analytics & Insights
- **Temporal Analysis**: Discover peak listening times, seasonal patterns, and long-term trends
- **Music Preferences**: Analyze listening patterns and artist preferences over time
- **Listening Behavior**: Track skip patterns, session duration, and platform usage evolution
- **Content Evolution**: Monitor shifts between music, podcasts, and other content types

### Interactive Dashboard
- **Overview Dashboard**: Key metrics with time-based filtering and content type breakdown
- **Temporal Patterns**: Visualize listening trends across hours, days, seasons, and years
- **Music Preferences**: Explore artist preferences and listening patterns
- **Listening Behavior**: Analyze session patterns, skip rates, and platform preferences
- **Deep Insights**: Advanced analytics with listening pattern analysis and trend detection
- **Memory-Optimized**: Efficient data enrichment processes for large datasets

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd Spotify-Analytics

# Run the setup script
python setup.py
```

The setup script will:
- Install all required dependencies
- Create necessary directories
- Create sample data structure
- Run basic tests

### 2. Add Your Personal Spotify Data

**Download your data from Spotify:**
1. Go to [Spotify Account Settings](https://www.spotify.com/account/)
2. Navigate to **Privacy Settings**
3. Click **Request data**
4. Download your **Account Data** and **Extended Streaming History**

Place your downloaded Spotify data files in `data/raw/`:

```
data/raw/
├── Streaming_History_Audio_2014-2021_0.json
├── Streaming_History_Audio_2021-2022_1.json
├── Streaming_History_Audio_2022-2023_2.json
├── Streaming_History_Video_2017-2025.json
├── StreamingHistory_podcast_0.json
└── ... (other JSON files from your Spotify export)
```

**Note**: The pipeline will work with any valid Spotify JSON export files from your personal account.

### 3. Run Analysis

```bash
# Run complete analysis pipeline
python main.py
```

This will:
- Parse your personal streaming history
- Create derived metrics
- Perform comprehensive analysis
- Generate insights

### 4. Launch Dashboard

```bash
# Launch interactive dashboard
streamlit run src/visualization/interactive_dashboard.py
```

The dashboard will open in your browser with:
- Overview metrics
- Temporal patterns
- Music preferences
- Listening behavior
- Deep insights

## What You'll Get

After running the pipeline, you'll have:

```
results/
├── key_insights.json          # Key findings
├── temporal_patterns_results.json
├── session_behavior_results.json
├── platform_usage_results.json
└── listening_efficiency_results.json

data/enriched/
├── enriched_streaming_data.parquet
├── daily_stats.parquet
├── weekly_stats.parquet
├── monthly_stats.parquet
├── artist_stats.parquet
└── summary_statistics.json
```

## Sample Insights You'll Discover

- **Peak listening times** and patterns
- **Favorite artists** and listening trends
- **Listening efficiency** and skip behavior
- **Platform preferences** over time
- **Seasonal trends** in music taste
- **Session behavior** and binge listening
- **Content type evolution** (music vs podcasts vs audiobooks)

## Technical Architecture

```
src/
├── data_processing/
│   ├── streaming_parser.py      # Parse JSON streaming history files
│   └── data_enrichment.py       # Create derived metrics and merge datasets
├── analysis/
│   └── listening_patterns.py    # Temporal, behavioral, and preference analysis
├── visualization/
│   └── interactive_dashboard.py # Streamlit dashboard with Plotly visualizations
└── ml_models/                   # Clustering and prediction models (future)

data/
├── raw/           # Original Spotify JSON exports (personal data)
├── processed/     # Parsed and cleaned data
└── enriched/      # Final analysis-ready datasets

results/           # Generated insights and analysis outputs
config/            # Settings and configuration
```

## Technologies

**Core Stack:**
- **Python** - Primary programming language
- **Pandas/NumPy** - Data manipulation and analysis
- **Streamlit** - Interactive web dashboard
- **Plotly** - Advanced data visualizations
- **Scikit-learn** - Machine learning models

**Data Sources:**
- **Spotify JSON Exports** - Personal streaming history (10+ years)

**Key Features:**
- **Scalable Data Processing** - Handles 100K+ records efficiently
- **Temporal Pattern Analysis** - Behavioral modeling and insights
- **Interactive Visualizations** - Data visualization for exploration
- **Memory Optimization** - Efficient data enrichment processes

## Advanced Usage

### Custom Analysis

```python
# Import components for custom analysis
from src.analysis.listening_patterns import ListeningPatternsAnalyzer
from src.visualization.interactive_dashboard import BeatScopeDashboard

# Load your data
import pandas as pd
df = pd.read_parquet("data/enriched/enriched_streaming_data.parquet")

# Run custom analysis
analyzer = ListeningPatternsAnalyzer()
results = analyzer.analyze_temporal_patterns(df)
```

### Export Results

```bash
# Export specific data
python -c "
import pandas as pd
df = pd.read_parquet('data/enriched/enriched_streaming_data.parquet')
df.to_csv('my_spotify_data.csv', index=False)
"
```

## Troubleshooting

### Common Issues

**"No data found" error:**
- Ensure JSON files are in `data/raw/`
- Check file names match expected patterns
- Verify JSON files are valid

**Memory issues:**
- Reduce batch size in `config/config.yaml`
- Process data in smaller chunks
- Use `--parse-only` then `--enrich-only` separately

### Getting Help

1. Check the logs: `logs/spotify_analytics.log`
2. Review configuration: `config/config.yaml`
3. Test individual components:
   ```bash
   python src/data_processing/streaming_parser.py
   ```

## Performance Tips

- **Large datasets**: Process in chunks with smaller batch sizes
- **Memory optimization**: Use appropriate batch sizes in configuration
- **Parallel processing**: Enabled by default for faster parsing

## Data Privacy & Security

- **Personal Project**: This project uses my own personal Spotify data
- **Local Processing**: All processing happens locally on your machine
- **No External Sharing**: No data is shared externally
- **Export Control**: Export functionality allows you to control what insights you share
- **Raw Data Only**: Works entirely with your exported data - no external dependencies
- **Git Protection**: Personal data files are excluded from version control via .gitignore

## Project Highlights

This **personal data-driven platform** demonstrates end-to-end data skills:
- **Data Engineering** - Large-scale JSON processing and data enrichment
- **Analytics** - Statistical analysis and behavioral insights
- **Visualization** - Interactive dashboards and storytelling with data
- **Machine Learning** - Clustering and predictive modeling
- **Software Engineering** - Clean architecture and modular design

**Skills Demonstrated:**
- **Python (Programming Language)** - Core development and data processing
- **Data Visualization** - Interactive dashboards and insights presentation

## Success!

You now have a comprehensive personal Spotify analytics system that works entirely with your raw data! 

**Next steps:**
- Explore the dashboard insights
- Customize analysis for your interests
- Build on the framework for new analyses

---

**Need help?** Check the logs and configuration files for detailed troubleshooting information.

