# type: ignore
"""
BeatScope: Spotify Analytics - Interactive Dashboard

This module creates a comprehensive Streamlit dashboard that visualizes
all aspects of the Spotify analytics, including temporal patterns,
music preferences, listening behavior, and insights.

⚠️  SECURITY NOTICE: This dashboard processes personal Spotify data.
    - All data processing happens locally
    - Personal data files are excluded from version control
    - No data is shared externally
    - Use only your own exported Spotify data
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import json
from datetime import datetime
import logging
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BeatScopeDashboard:
    """
    Comprehensive Streamlit dashboard for BeatScope: Spotify Analytics
    
    Features:
    - Overview dashboard with key metrics
    - Temporal analysis visualizations
    - Music preference analysis
    - Listening behavior insights
    - Interactive filters and controls
    - Export functionality
    """
    
    def __init__(self, data_path: str = "data/enriched") -> None:
        """
        Initialize the dashboard
        
        Args:
            data_path: Path to enriched data directory
        """
        self.data_path: Path = Path(data_path)
        self.data: pd.DataFrame = pd.DataFrame()
        self.filtered_data: pd.DataFrame = pd.DataFrame()
        self.aggregated_data: Dict[str, pd.DataFrame] = {}
        self.summary_stats: Dict[str, Any] = {}
        self.load_data()
        # Initialize filtered_data with all data
        if not self.data.empty:  # type: ignore
            self.filtered_data = self.data.copy()
        
    def load_data(self):
        """Load all necessary data files"""
        try:
            # Load main enriched data
            main_file = self.data_path / "enriched_streaming_data.parquet"
            if main_file.exists():
                self.data = pd.read_parquet(main_file)
                logger.info(f"Loaded {len(self.data)} records from enriched data")
            else:
                logger.warning("Enriched data not found. Please run data processing first.")
                self.data = pd.DataFrame()
            
            # Load aggregated data
            self.aggregated_data = {}
            for agg_file in self.data_path.glob("*.parquet"):
                if agg_file.name != "enriched_streaming_data.parquet":
                    name = agg_file.stem
                    self.aggregated_data[name] = pd.read_parquet(agg_file)
            
            # Load summary statistics
            summary_file = self.data_path / "summary_statistics.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    self.summary_stats = json.load(f)
            else:
                self.summary_stats = {}
                
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            self.data = pd.DataFrame()
            self.aggregated_data = {}
            self.summary_stats = {}
    
    def run_dashboard(self):
        """Run the main dashboard application"""
        st.set_page_config(
            page_title="BeatScope: Spotify Analytics",
            page_icon="",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for streamlined UI
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1DB954;
            text-align: left;
            margin-bottom: 1.5rem;
            font-weight: 600;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #1DB954;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .insight-box {
            background-color: #f0f8f0;
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid #1DB954;
            margin-bottom: 1rem;
        }
        .sidebar-header {
            font-size: 1.2rem;
            font-weight: 600;
            color: #1DB954;
            margin-bottom: 0.5rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f8f9fa;
            border-radius: 4px 4px 0px 0px;
            color: #495057;
            padding: 10px 16px;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1DB954;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown('<h1 class="main-header">BeatScope: Spotify Analytics</h1>', unsafe_allow_html=True)
        
        if self.data.empty:
            st.error("No data found. Please ensure you have processed your Spotify data first.")
            st.info("Run the data processing pipeline to generate the required files.")
            return
        

        
        # Sidebar
        self.create_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs([
            "Overview",
            "Temporal Patterns",
            "Listening Behavior",
            "Deep Insights"
        ])
        
        with tab1:
            self.show_overview()
        
        with tab2:
            self.show_temporal_patterns()
        
        with tab3:
            self.show_listening_behavior()
        
        with tab4:
            self.show_deep_insights()
    
    def create_sidebar(self):
        """Create streamlined sidebar with essential filters"""
        st.sidebar.title("Filters")
        
        if self.data.empty:
            st.sidebar.warning("No data available")
            self.filtered_data = self.data
            return
        
        # Initialize filtered data
        self.filtered_data = self.data.copy()
        
        # Date range filter
        min_date = self.data['timestamp'].min().date()
        max_date = self.data['timestamp'].max().date()
        
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            self.filtered_data = self.filtered_data[
                (self.filtered_data['timestamp'].dt.date >= start_date) &
                (self.filtered_data['timestamp'].dt.date <= end_date)
            ]
        
        # Filters in expandable sections for cleaner UI
        with st.sidebar.expander("Content & Platform", expanded=False):
            # Content type filter
            content_types = ['All'] + sorted(self.data['content_type'].unique())
            selected_content = st.selectbox("Content Type", content_types, index=0)
            
            if selected_content != 'All':
                self.filtered_data = self.filtered_data[
                    self.filtered_data['content_type'] == selected_content
                ]
            
            # Platform filter
            if 'platform_category' in self.data.columns:
                platforms = ['All'] + sorted(self.data['platform_category'].unique())
                selected_platform = st.selectbox("Platform", platforms, index=0)
                
                if selected_platform != 'All':
                    self.filtered_data = self.filtered_data[
                        self.filtered_data['platform_category'] == selected_platform
                    ]
        
        # Quick stats
        if not self.filtered_data.empty:  # type: ignore
            st.sidebar.markdown("---")
            st.sidebar.subheader("Quick Stats")
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                total_hours = self.filtered_data['ms_played'].sum() / 3600000
                st.metric("Hours", f"{total_hours:.1f}")
                
            with col2:
                total_tracks = len(self.filtered_data)
                st.metric("Tracks", f"{total_tracks:,}")
            
            unique_artists = self.filtered_data['artist_name'].nunique()
            st.sidebar.metric("Artists", f"{unique_artists:,}")
    
    def show_overview(self):
        """Show overview dashboard with key metrics"""
        st.header("Overview Dashboard")
        
        if self.filtered_data.empty:
            st.warning("No data available for the selected filters.")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_hours = self.filtered_data['ms_played'].sum() / 3600000
            st.metric("Total Listening Hours", f"{total_hours:.1f}")
        
        with col2:
            total_tracks = len(self.filtered_data)
            st.metric("Total Tracks Played", f"{total_tracks:,}")
        
        with col3:
            unique_artists = self.filtered_data['artist_name'].nunique()
            st.metric("Unique Artists", f"{unique_artists:,}")
        
        with col4:
            avg_efficiency = self.filtered_data['listening_efficiency'].mean()
            st.metric("Avg. Listening Efficiency", f"{avg_efficiency:.1%}")
        
        # Content type distribution
        st.subheader("Content Type Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            content_dist = self.filtered_data['content_type'].value_counts()
            fig = px.pie(
                values=content_dist.values,
                names=content_dist.index,
                title="Content Type Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            content_hours = self.filtered_data.groupby('content_type')['ms_played'].sum() / 3600000
            fig = px.bar(
                x=content_hours.index,
                y=content_hours.values,
                title="Listening Hours by Content Type",
                labels={'x': 'Content Type', 'y': 'Hours'},
                color=content_hours.values,
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top artists
        st.subheader("Top Artists")
        top_artists = self.filtered_data.groupby('artist_name', observed=False).agg({
            'ms_played': 'sum',
            'track_name': 'count'
        }).reset_index()
        
        top_artists['listening_hours'] = top_artists['ms_played'] / 3600000
        top_artists = top_artists.nlargest(10, 'listening_hours')
        
        fig = px.bar(
            top_artists,
            x='listening_hours',
            y='artist_name',
            orientation='h',
            title="Top 10 Artists by Listening Hours",
            labels={'listening_hours': 'Hours', 'artist_name': 'Artist'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Platform usage
        st.subheader("Platform Usage")
        platform_usage = self.filtered_data['platform_category'].value_counts()
        
        fig = px.pie(
            values=platform_usage.values,
            names=platform_usage.index,
            title="Platform Usage Distribution",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def show_temporal_patterns(self):
        """Show temporal pattern analysis"""
        st.header("Temporal Patterns")
        
        if self.filtered_data.empty:
            st.warning("No data available for the selected filters.")
            return
        
        # Daily patterns
        st.subheader("Daily Patterns")
        
        # Create date column if not exists
        if 'date' not in self.filtered_data.columns:
            self.filtered_data['date'] = self.filtered_data['timestamp'].dt.date
        
        daily_stats = self.filtered_data.groupby(['date', 'content_type'], observed=False).agg({
            'ms_played': 'sum',
            'track_name': 'count'
        }).reset_index()
        
        daily_stats['listening_hours'] = daily_stats['ms_played'] / 3600000
        
        fig = px.line(
            daily_stats,
            x='date',
            y='listening_hours',
            color='content_type',
            title="Daily Listening Hours Over Time",
            labels={'listening_hours': 'Hours', 'date': 'Date'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Time of day patterns
        st.subheader("Time of Day Patterns")
        col1, col2 = st.columns(2)
        
        with col1:
            time_stats = self.filtered_data.groupby('time_of_day', observed=False).agg({
                'ms_played': 'sum',
                'track_name': 'count'
            }).reset_index()
            
            time_stats['listening_hours'] = time_stats['ms_played'] / 3600000
            
            fig = px.bar(
                time_stats,
                x='time_of_day',
                y='listening_hours',
                title="Listening Hours by Time of Day",
                labels={'listening_hours': 'Hours', 'time_of_day': 'Time of Day'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Hourly heatmap
            hourly_data = self.filtered_data.groupby(['hour', 'day_of_week'], observed=False).agg({
                'ms_played': 'sum'
            }).reset_index()
            
            hourly_data['listening_hours'] = hourly_data['ms_played'] / 3600000
            
            # Pivot for heatmap
            heatmap_data = hourly_data.pivot(
                index='day_of_week', 
                columns='hour', 
                values='listening_hours'
            ).fillna(0)
            
            fig = px.imshow(
                heatmap_data,
                title="Listening Hours Heatmap (Day vs Hour)",
                labels=dict(x="Hour of Day", y="Day of Week", color="Hours"),
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Weekly patterns
        st.subheader("Weekly Patterns")
        weekly_stats = self.filtered_data.groupby(['day_of_week', 'content_type'], observed=False).agg({
            'ms_played': 'sum',
            'track_name': 'count'
        }).reset_index()
        
        weekly_stats['listening_hours'] = weekly_stats['ms_played'] / 3600000
        
        fig = px.bar(
            weekly_stats,
            x='day_of_week',
            y='listening_hours',
            color='content_type',
            title="Average Listening Hours by Day of Week",
            labels={'listening_hours': 'Hours', 'day_of_week': 'Day of Week'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal patterns
        st.subheader("Seasonal Patterns")
        seasonal_stats = self.filtered_data.groupby(['season', 'content_type'], observed=False).agg({
            'ms_played': 'sum',
            'track_name': 'count'
        }).reset_index()
        
        seasonal_stats['listening_hours'] = seasonal_stats['ms_played'] / 3600000
        
        fig = px.bar(
            seasonal_stats,
            x='season',
            y='listening_hours',
            color='content_type',
            title="Listening Hours by Season",
            labels={'listening_hours': 'Hours', 'season': 'Season'}
        )
        st.plotly_chart(fig, use_container_width=True)
    

    
    def show_listening_behavior(self):
        """Show listening behavior analysis"""
        st.header("Listening Behavior")
        
        if self.filtered_data.empty:
            st.warning("No data available for the selected filters.")
            return
        
        # Session analysis
        st.subheader("Session Analysis")
        
        if 'session_duration_minutes' in self.filtered_data.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                session_duration = self.filtered_data['session_duration_minutes'].dropna()
                fig = px.histogram(
                    session_duration,
                    title="Session Duration Distribution",
                    labels={'value': 'Duration (minutes)', 'count': 'Number of Sessions'},
                    nbins=30
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Session duration by content type
                session_by_type = self.filtered_data.groupby('content_type')['session_duration_minutes'].mean()
                fig = px.bar(
                    x=session_by_type.index,
                    y=session_by_type.values,
                    title="Average Session Duration by Content Type",
                    labels={'x': 'Content Type', 'y': 'Duration (minutes)'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Listening efficiency
        st.subheader("Listening Efficiency")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'listening_efficiency' in self.filtered_data.columns:
                efficiency_dist = self.filtered_data['listening_efficiency'].dropna()
                fig = px.histogram(
                    efficiency_dist,
                    title="Listening Efficiency Distribution",
                    labels={'value': 'Efficiency', 'count': 'Number of Tracks'},
                    nbins=30
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Efficiency by platform
            if 'listening_efficiency' in self.filtered_data.columns and 'platform_category' in self.filtered_data.columns:
                platform_efficiency = self.filtered_data.groupby('platform_category')['listening_efficiency'].mean()
                fig = px.bar(
                    x=platform_efficiency.index,
                    y=platform_efficiency.values,
                    title="Average Listening Efficiency by Platform",
                    labels={'x': 'Platform', 'y': 'Efficiency'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Skip behavior
        st.subheader("Skip Behavior")
        if 'is_skip' in self.filtered_data.columns:
            skip_stats = self.filtered_data.groupby(['content_type', 'time_of_day'], observed=False).agg({
                'is_skip': 'sum',
                'track_name': 'count'
            }).reset_index()
            
            skip_stats['skip_rate'] = skip_stats['is_skip'] / skip_stats['track_name']
            
            fig = px.bar(
                skip_stats,
                x='time_of_day',
                y='skip_rate',
                color='content_type',
                title="Skip Rate by Time of Day and Content Type",
                labels={'skip_rate': 'Skip Rate', 'time_of_day': 'Time of Day'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Platform evolution
        st.subheader("Platform Evolution Over Time")
        if 'platform_category' in self.filtered_data.columns:
            platform_evolution = self.filtered_data.groupby(['year', 'platform_category'], observed=False).agg({
                'ms_played': 'sum'
            }).reset_index()
            
            platform_evolution['listening_hours'] = platform_evolution['ms_played'] / 3600000
            
            fig = px.line(
                platform_evolution,
                x='year',
                y='listening_hours',
                color='platform_category',
                title="Platform Usage Evolution",
                labels={'listening_hours': 'Hours', 'year': 'Year'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def show_deep_insights(self):
        """Show deep insights and advanced analytics"""
        st.header("Deep Insights")
        
        if self.filtered_data.empty:
            st.warning("No data available for the selected filters.")
            return
        
        # Key insights
        st.subheader("Key Insights")
        
        insights = []
        
        # Total listening time
        total_hours = self.filtered_data['ms_played'].sum() / 3600000
        insights.append(f"**Total Listening Time**: {total_hours:.1f} hours")
        
        # Most listened artist
        top_artist = self.filtered_data.groupby('artist_name')['ms_played'].sum().idxmax()
        artist_hours = self.filtered_data.groupby('artist_name')['ms_played'].sum().max() / 3600000
        insights.append(f"**Top Artist**: {top_artist} ({artist_hours:.1f} hours)")
        
        # Peak listening time
        if 'time_of_day' in self.filtered_data.columns:
            peak_time = self.filtered_data.groupby('time_of_day')['ms_played'].sum().idxmax()
            insights.append(f"**Peak Listening Time**: {peak_time}")
        
        # Favorite season
        if 'season' in self.filtered_data.columns:
            favorite_season = self.filtered_data.groupby('season')['ms_played'].sum().idxmax()
            insights.append(f"**Favorite Season**: {favorite_season}")
        
        # Listening efficiency
        if 'listening_efficiency' in self.filtered_data.columns:
            avg_efficiency = self.filtered_data['listening_efficiency'].mean()
            insights.append(f"**Average Listening Efficiency**: {avg_efficiency:.1%}")
        
        # Display insights
        for insight in insights:
            st.markdown(f"• {insight}")
        
        # Advanced analytics
        st.subheader("Advanced Analytics")
        
        # Listening pattern analysis
        st.subheader("Listening Pattern Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hour of day analysis
            if 'hour' in self.filtered_data.columns:
                hour_dist = self.filtered_data['hour'].value_counts().sort_index()
                fig = px.bar(
                    x=hour_dist.index,
                    y=hour_dist.values,
                    title="Listening Activity by Hour of Day",
                    labels={'x': 'Hour', 'y': 'Number of Tracks'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Day of week analysis
            if 'day_of_week' in self.filtered_data.columns:
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_dist = self.filtered_data['day_of_week'].value_counts().sort_index()
                day_dist.index = [day_names[i] for i in day_dist.index]
                
                fig = px.bar(
                    x=day_dist.index,
                    y=day_dist.values,
                    title="Listening Activity by Day of Week",
                    labels={'x': 'Day', 'y': 'Number of Tracks'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Trend analysis
        st.subheader("Trend Analysis")
        
        if 'year' in self.filtered_data.columns:
            yearly_trends = self.filtered_data.groupby(['year', 'content_type'], observed=False).agg({
                'ms_played': 'sum',
                'track_name': 'count'
            }).reset_index()
            
            yearly_trends['listening_hours'] = yearly_trends['ms_played'] / 3600000
            
            fig = px.line(
                yearly_trends,
                x='year',
                y='listening_hours',
                color='content_type',
                title="Yearly Listening Trends",
                labels={'listening_hours': 'Hours', 'year': 'Year'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Export functionality
        st.subheader("Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Filtered Data"):
                csv = self.filtered_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"spotify_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Export Summary Report"):
                # Create summary report
                report = self._create_summary_report()
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"spotify_summary_report_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
    
    def _create_summary_report(self) -> str:
        """Create a summary report of the analytics"""
        report_lines = []
        report_lines.append("=" * 50)
        report_lines.append("SPOTIFY ANALYTICS SUMMARY REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        if not self.filtered_data.empty:
            # Basic stats
            total_hours = self.filtered_data['ms_played'].sum() / 3600000
            total_tracks = len(self.filtered_data)
            unique_artists = self.filtered_data['artist_name'].nunique()
            
            report_lines.append("BASIC STATISTICS:")
            report_lines.append(f"- Total Listening Hours: {total_hours:.1f}")
            report_lines.append(f"- Total Tracks Played: {total_tracks:,}")
            report_lines.append(f"- Unique Artists: {unique_artists:,}")
            report_lines.append("")
            
            # Content type breakdown
            content_breakdown = self.filtered_data['content_type'].value_counts()
            report_lines.append("CONTENT TYPE BREAKDOWN:")
            for content_type, count in content_breakdown.items():
                percentage = (count / total_tracks) * 100
                report_lines.append(f"- {content_type}: {count:,} tracks ({percentage:.1f}%)")
            report_lines.append("")
            # Top artists
            if 'artist_name' in self.filtered_data.columns and 'ms_played' in self.filtered_data.columns:
                top_artists = (
                    self.filtered_data.groupby('artist_name', as_index=False)['ms_played']
                    .sum()
                    .sort_values('ms_played', ascending=False)
                    .head(5)
                )
                report_lines.append("TOP 5 ARTISTS:")
                for i, row in enumerate(top_artists.itertuples(index=False), 1):
                    hours = row.ms_played / 3600000
                    report_lines.append(f"{i}. {row.artist_name}: {hours:.1f} hours")
            else:
                report_lines.append("TOP 5 ARTISTS: Data not available.")
            report_lines.append("")
            
            # Platform usage
            if 'platform_category' in self.filtered_data.columns:
                platform_usage = self.filtered_data['platform_category'].value_counts()
                report_lines.append("PLATFORM USAGE:")
                for platform, count in platform_usage.items():
                    percentage = (count / total_tracks) * 100
                    report_lines.append(f"- {platform}: {count:,} tracks ({percentage:.1f}%)")
                report_lines.append("")
        
        return "\n".join(report_lines)


def main():
    """Main function to run the dashboard"""
    dashboard = BeatScopeDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()
