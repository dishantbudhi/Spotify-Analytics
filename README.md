# Spotify Analytics Platform

**A data engineering project analyzing 10+ years of personal streaming data (100K+ records) to extract behavioral insights through advanced temporal analysis and interactive visualizations.**

## 🎯 Project Highlights

- Processes **100,000+ streaming records** spanning 10+ years from Spotify's extended data export
- Built complete **ETL pipeline** with Python for multi-format JSON parsing and data enrichment
- Implements **temporal pattern analysis** revealing hourly, daily, seasonal, and yearly trends
- Features **interactive Streamlit dashboard** with real-time filtering and data export capabilities
- Maintains **100% data privacy** with local-only processing

## 🛠 Technical Stack

**Core:** Python, Pandas, NumPy  
**Visualization:** Streamlit, Plotly  
**Analytics:** Scikit-learn (clustering, behavioral modeling)  
**Data:** Parquet, JSON  

## 📊 Key Features

### Data Pipeline
- Multi-format parser handling music, podcasts, audiobooks, and video content
- Batch processing optimization for large-scale JSON datasets
- Derived metrics: skip rates, listening efficiency, session patterns
- Automated data enrichment and aggregation

### Analytics Engine
- **Temporal Analysis:** Peak listening times, seasonal patterns, multi-year trends
- **Behavioral Modeling:** Skip patterns, session duration distributions, platform usage
- **Preference Evolution:** Artist and genre shifts over time
- **Content Analysis:** Format preferences (music vs. podcasts) and consumption patterns

### Interactive Dashboard
- Real-time filtering across time ranges and content types
- Exportable visualizations and processed datasets
- Session analysis with efficiency metrics
- Platform usage breakdown (mobile, desktop, smart speakers)

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd spotify-analytics
python setup.py

# Add your Spotify data export to data/raw/
# Download from: spotify.com/account/privacy

# Run analysis pipeline
python main.py

# Launch dashboard
streamlit run src/visualization/interactive_dashboard.py
```

## 📁 Architecture

```
src/
├── data_processing/      # ETL pipeline & parsers
├── analysis/            # Pattern detection & modeling  
├── visualization/       # Dashboard components
└── config/             # Batch sizes & parameters

results/
├── enriched_data.parquet       # Processed dataset
├── temporal_patterns.json      # Time-based insights
└── session_behavior.json       # Behavioral metrics
```

## 💡 Sample Insights Generated

- Identified optimal productivity hours through listening pattern analysis
- Discovered 3 distinct listening "personas" through clustering
- Quantified attention span changes over 10-year period
- Revealed seasonal genre preferences and mood patterns

## 🔒 Privacy Note

This project exclusively analyzes my personal data, processed entirely locally. No external APIs or third-party services are used.

## 📈 Skills Demonstrated

- **Data Engineering:** Large-scale JSON processing, ETL pipeline design
- **Analytics:** Temporal analysis, behavioral modeling, pattern recognition
- **Software Architecture:** Modular design, configuration management
- **Visualization:** Interactive dashboards, real-time filtering
