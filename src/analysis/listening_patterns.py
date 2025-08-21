"""
Listening Patterns Analysis for Spotify Personal Analytics Engine

This module analyzes temporal patterns, session behavior, and listening habits
from the enriched streaming data to uncover deep insights about music consumption.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ListeningPatternsAnalyzer:
    """
    Comprehensive analyzer for listening patterns and temporal behavior
    
    Features:
    - Temporal pattern analysis (daily, weekly, monthly, seasonal)
    - Session behavior analysis
    - Platform usage patterns
    - Listening efficiency analysis
    - Content type preferences over time
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize listening patterns analyzer
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.results = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze temporal patterns in listening behavior
        
        Args:
            df: Enriched streaming DataFrame
            
        Returns:
            Dictionary with temporal analysis results
        """
        logger.info("Analyzing temporal patterns...")
        
        results = {}
        
        # Daily patterns
        results['daily_patterns'] = self._analyze_daily_patterns(df)
        
        # Weekly patterns
        results['weekly_patterns'] = self._analyze_weekly_patterns(df)
        
        # Monthly patterns
        results['monthly_patterns'] = self._analyze_monthly_patterns(df)
        
        # Seasonal patterns
        results['seasonal_patterns'] = self._analyze_seasonal_patterns(df)
        
        # Time of day patterns
        results['time_of_day_patterns'] = self._analyze_time_of_day_patterns(df)
        
        # Year-over-year trends
        results['yearly_trends'] = self._analyze_yearly_trends(df)
        
        self.results['temporal_patterns'] = results
        logger.info("Temporal patterns analysis completed")
        
        return results
    
    def _analyze_daily_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze daily listening patterns"""
        daily_stats = df.groupby(['date', 'content_type'], observed=False).agg({
            'ms_played': 'sum',
            'track_name': 'count',
            'is_complete_listen': 'sum',
            'is_skip': 'sum',
            'valence': 'mean',
            'energy': 'mean',
            'danceability': 'mean',
            'tempo': 'mean'
        }).reset_index()
        
        daily_stats['listening_hours'] = daily_stats['ms_played'] / 3600000
        daily_stats['skip_rate'] = daily_stats['is_skip'] / daily_stats['track_name']
        daily_stats['completion_rate'] = daily_stats['is_complete_listen'] / daily_stats['track_name']
        
        # Day of week analysis
        daily_stats['day_of_week'] = pd.to_datetime(daily_stats['date']).dt.dayofweek
        day_of_week_stats = daily_stats.groupby(['day_of_week', 'content_type'], observed=False).agg({
            'listening_hours': 'mean',
            'track_name': 'mean',
            'skip_rate': 'mean',
            'completion_rate': 'mean',
            'valence': 'mean',
            'energy': 'mean'
        }).reset_index()
        
        return {
            'daily_stats': daily_stats,
            'day_of_week_stats': day_of_week_stats,
            'total_days': daily_stats['date'].nunique(),
            'average_daily_hours': daily_stats.groupby('content_type')['listening_hours'].mean().to_dict(),
            'peak_day': day_of_week_stats.loc[day_of_week_stats['listening_hours'].idxmax()].to_dict()
        }
    
    def _analyze_weekly_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze weekly listening patterns"""
        weekly_stats = df.groupby(['year', 'week_of_year', 'content_type'], observed=False).agg({
            'ms_played': 'sum',
            'track_name': 'count',
            'artist_name': 'nunique',
            'album_name': 'nunique',
            'valence': 'mean',
            'energy': 'mean',
            'danceability': 'mean',
            'tempo': 'mean',
            'session_id': 'nunique'
        }).reset_index()
        
        weekly_stats['listening_hours'] = weekly_stats['ms_played'] / 3600000
        weekly_stats['unique_artists'] = weekly_stats['artist_name']
        weekly_stats['unique_albums'] = weekly_stats['album_name']
        weekly_stats['total_sessions'] = weekly_stats['session_id']
        
        # Weekly trends
        weekly_trends = weekly_stats.groupby('content_type').agg({
            'listening_hours': ['mean', 'std', 'min', 'max'],
            'unique_artists': 'mean',
            'total_sessions': 'mean'
        }).round(2)
        
        return {
            'weekly_stats': weekly_stats,
            'weekly_trends': weekly_trends,
            'total_weeks': weekly_stats['week_of_year'].nunique(),
            'average_weekly_hours': weekly_stats.groupby('content_type')['listening_hours'].mean().to_dict()
        }
    
    def _analyze_monthly_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze monthly listening patterns"""
        monthly_stats = df.groupby(['year', 'month', 'content_type'], observed=False).agg({
            'ms_played': 'sum',
            'track_name': 'count',
            'artist_name': 'nunique',
            'album_name': 'nunique',
            'valence': 'mean',
            'energy': 'mean',
            'danceability': 'mean',
            'tempo': 'mean',
            'session_id': 'nunique'
        }).reset_index()
        
        monthly_stats['listening_hours'] = monthly_stats['ms_played'] / 3600000
        monthly_stats['unique_artists'] = monthly_stats['artist_name']
        monthly_stats['unique_albums'] = monthly_stats['album_name']
        monthly_stats['total_sessions'] = monthly_stats['session_id']
        
        # Monthly trends and seasonality
        monthly_trends = monthly_stats.groupby('content_type').agg({
            'listening_hours': ['mean', 'std', 'min', 'max'],
            'unique_artists': 'mean',
            'total_sessions': 'mean'
        }).round(2)
        
        # Peak months
        peak_months = monthly_stats.loc[monthly_stats.groupby('content_type')['listening_hours'].idxmax()]
        
        return {
            'monthly_stats': monthly_stats,
            'monthly_trends': monthly_trends,
            'peak_months': peak_months.to_dict('records'),
            'total_months': monthly_stats['month'].nunique()
        }
    
    def _analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze seasonal patterns in listening behavior"""
        seasonal_stats = df.groupby(['season', 'content_type'], observed=False).agg({
            'ms_played': 'sum',
            'track_name': 'count',
            'artist_name': 'nunique',
            'valence': 'mean',
            'energy': 'mean',
            'danceability': 'mean',
            'tempo': 'mean'
        }).reset_index()
        
        seasonal_stats['listening_hours'] = seasonal_stats['ms_played'] / 3600000
        seasonal_stats['unique_artists'] = seasonal_stats['artist_name']
        
        # Seasonal preferences
        seasonal_preferences = seasonal_stats.groupby('content_type').apply(
            lambda x: x.loc[x['listening_hours'].idxmax(), 'season']
        ).to_dict()
        
        # Audio feature changes by season
        seasonal_audio_features = seasonal_stats.groupby('season', observed=False).agg({
            'valence': 'mean',
            'energy': 'mean',
            'danceability': 'mean',
            'tempo': 'mean'
        }).round(3)
        
        return {
            'seasonal_stats': seasonal_stats,
            'seasonal_preferences': seasonal_preferences,
            'seasonal_audio_features': seasonal_audio_features,
            'favorite_season': seasonal_stats.loc[seasonal_stats['listening_hours'].idxmax(), 'season']
        }
    
    def _analyze_time_of_day_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze time of day listening patterns"""
        time_stats = df.groupby(['time_of_day', 'content_type'], observed=False).agg({
            'ms_played': 'sum',
            'track_name': 'count',
            'valence': 'mean',
            'energy': 'mean',
            'danceability': 'mean',
            'tempo': 'mean'
        }).reset_index()
        
        time_stats['listening_hours'] = time_stats['ms_played'] / 3600000
        
        # Peak listening times
        peak_times = time_stats.loc[time_stats.groupby('content_type')['listening_hours'].idxmax()]
        
        # Audio feature patterns by time of day
        time_audio_features = time_stats.groupby('time_of_day', observed=False).agg({
            'valence': 'mean',
            'energy': 'mean',
            'danceability': 'mean',
            'tempo': 'mean'
        }).round(3)
        
        return {
            'time_stats': time_stats,
            'peak_times': peak_times.to_dict('records'),
            'time_audio_features': time_audio_features,
            'favorite_time': time_stats.loc[time_stats['listening_hours'].idxmax(), 'time_of_day']
        }
    
    def _analyze_yearly_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze year-over-year trends"""
        yearly_stats = df.groupby(['year', 'content_type'], observed=False).agg({
            'ms_played': 'sum',
            'track_name': 'count',
            'artist_name': 'nunique',
            'album_name': 'nunique',
            'valence': 'mean',
            'energy': 'mean',
            'danceability': 'mean',
            'tempo': 'mean'
        }).reset_index()
        
        yearly_stats['listening_hours'] = yearly_stats['ms_played'] / 3600000
        yearly_stats['unique_artists'] = yearly_stats['artist_name']
        yearly_stats['unique_albums'] = yearly_stats['album_name']
        
        # Year-over-year growth
        yearly_growth = yearly_stats.groupby('content_type').apply(
            lambda x: x.sort_values('year').assign(
                growth_rate=lambda y: y['listening_hours'].pct_change() * 100
            )
        ).reset_index(drop=True)
        
        # Overall trends
        overall_trends = yearly_stats.groupby('year', observed=False).agg({
            'listening_hours': 'sum',
            'unique_artists': 'sum',
            'valence': 'mean',
            'energy': 'mean'
        }).round(2)
        
        return {
            'yearly_stats': yearly_stats,
            'yearly_growth': yearly_growth,
            'overall_trends': overall_trends,
            'total_years': yearly_stats['year'].nunique(),
            'growth_rate': yearly_growth['growth_rate'].mean()
        }
    
    def analyze_session_behavior(self, df: pd.DataFrame) -> Dict:
        """
        Analyze session behavior and listening patterns
        
        Args:
            df: Enriched streaming DataFrame
            
        Returns:
            Dictionary with session analysis results
        """
        logger.info("Analyzing session behavior...")
        
        results = {}
        
        # Session statistics
        session_stats = df.groupby(['content_type', 'session_id'], observed=False).agg({
            'ms_played': 'sum',
            'track_name': 'count',
            'timestamp': ['min', 'max'],
            'valence': 'mean',
            'energy': 'mean',
            'danceability': 'mean',
            'is_skip': 'sum',
            'is_complete_listen': 'sum'
        }).reset_index()
        
        session_stats.columns = ['content_type', 'session_id', 'session_duration', 'tracks_in_session', 
                               'session_start', 'session_end', 'avg_valence', 'avg_energy', 
                               'avg_danceability', 'skips', 'completes']
        
        session_stats['session_duration_minutes'] = session_stats['session_duration'] / 60000
        session_stats['skip_rate'] = session_stats['skips'] / session_stats['tracks_in_session']
        session_stats['completion_rate'] = session_stats['completes'] / session_stats['tracks_in_session']
        
        # Session patterns by content type
        session_patterns = session_stats.groupby('content_type', observed=False).agg({
            'session_duration_minutes': ['mean', 'median', 'std', 'min', 'max'],
            'tracks_in_session': ['mean', 'median', 'std'],
            'skip_rate': 'mean',
            'completion_rate': 'mean'
        }).round(2)
        
        # Session clustering
        session_clusters = self._cluster_sessions(session_stats)
        
        # Binge listening detection
        binge_sessions = session_stats[session_stats['session_duration_minutes'] > 60]
        
        results = {
            'session_stats': session_stats,
            'session_patterns': session_patterns,
            'session_clusters': session_clusters,
            'binge_sessions': {
                'count': len(binge_sessions),
                'percentage': len(binge_sessions) / len(session_stats) * 100,
                'average_duration': binge_sessions['session_duration_minutes'].mean()
            },
            'total_sessions': len(session_stats),
            'average_session_duration': session_stats['session_duration_minutes'].mean()
        }
        
        self.results['session_behavior'] = results
        logger.info("Session behavior analysis completed")
        
        return results
    
    def _cluster_sessions(self, session_stats: pd.DataFrame) -> Dict:
        """Cluster sessions based on duration and behavior"""
        # Prepare features for clustering
        features = ['session_duration_minutes', 'tracks_in_session', 'skip_rate', 'completion_rate']
        X = session_stats[features].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        session_stats['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Analyze clusters
        cluster_analysis = session_stats.groupby('cluster', observed=False).agg({
            'session_duration_minutes': 'mean',
            'tracks_in_session': 'mean',
            'skip_rate': 'mean',
            'completion_rate': 'mean',
            'content_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
        }).round(2)
        
        # Label clusters
        cluster_labels = {
            0: 'Short Sessions',
            1: 'Long Sessions',
            2: 'High Skip Rate',
            3: 'Engaged Listening'
        }
        
        cluster_analysis['cluster_type'] = cluster_analysis.index.map(cluster_labels)
        
        return {
            'cluster_analysis': cluster_analysis,
            'cluster_labels': cluster_labels,
            'cluster_distribution': session_stats['cluster'].value_counts().to_dict()
        }
    
    def analyze_platform_usage(self, df: pd.DataFrame) -> Dict:
        """
        Analyze platform usage patterns
        
        Args:
            df: Enriched streaming DataFrame
            
        Returns:
            Dictionary with platform analysis results
        """
        logger.info("Analyzing platform usage patterns...")
        
        platform_stats = df.groupby(['platform_category', 'content_type'], observed=False).agg({
            'ms_played': 'sum',
            'track_name': 'count',
            'is_skip': 'sum',
            'is_complete_listen': 'sum',
            'valence': 'mean',
            'energy': 'mean',
            'danceability': 'mean',
            'session_id': 'nunique'
        }).reset_index()
        
        platform_stats['listening_hours'] = platform_stats['ms_played'] / 3600000
        platform_stats['skip_rate'] = platform_stats['is_skip'] / platform_stats['track_name']
        platform_stats['completion_rate'] = platform_stats['is_complete_listen'] / platform_stats['track_name']
        platform_stats['total_sessions'] = platform_stats['session_id']
        
        # Platform preferences
        platform_preferences = platform_stats.groupby('content_type').apply(
            lambda x: x.loc[x['listening_hours'].idxmax(), 'platform_category']
        ).to_dict()
        
        # Platform evolution over time
        platform_evolution = df.groupby(['year', 'platform_category'], observed=False).agg({
            'ms_played': 'sum',
            'track_name': 'count'
        }).reset_index()
        
        platform_evolution['listening_hours'] = platform_evolution['ms_played'] / 3600000
        
        results = {
            'platform_stats': platform_stats,
            'platform_preferences': platform_preferences,
            'platform_evolution': platform_evolution,
            'total_platforms': platform_stats['platform_category'].nunique(),
            'most_used_platform': platform_stats.loc[platform_stats['listening_hours'].idxmax(), 'platform_category']
        }
        
        self.results['platform_usage'] = results
        logger.info("Platform usage analysis completed")
        
        return results
    
    def analyze_listening_efficiency(self, df: pd.DataFrame) -> Dict:
        """
        Analyze listening efficiency and skip patterns
        
        Args:
            df: Enriched streaming DataFrame
            
        Returns:
            Dictionary with efficiency analysis results
        """
        logger.info("Analyzing listening efficiency...")
        
        # Overall efficiency metrics
        efficiency_stats = df.groupby('content_type', observed=False).agg({
            'listening_efficiency': ['mean', 'median', 'std'],
            'is_skip': 'sum',
            'is_complete_listen': 'sum',
            'track_name': 'count'
        }).round(3)
        
        efficiency_stats.columns = ['avg_efficiency', 'median_efficiency', 'std_efficiency', 
                                  'total_skips', 'total_completes', 'total_tracks']
        efficiency_stats['skip_rate'] = efficiency_stats['total_skips'] / efficiency_stats['total_tracks']
        efficiency_stats['completion_rate'] = efficiency_stats['total_completes'] / efficiency_stats['total_tracks']
        
        # Efficiency by time of day
        time_efficiency = df.groupby(['time_of_day', 'content_type'], observed=False).agg({
            'listening_efficiency': 'mean',
            'is_skip': 'sum',
            'track_name': 'count'
        }).reset_index()
        
        time_efficiency['skip_rate'] = time_efficiency['is_skip'] / time_efficiency['track_name']
        
        # Efficiency by audio features
        feature_efficiency = df.groupby('mood', observed=False).agg({
            'listening_efficiency': 'mean',
            'is_skip': 'sum',
            'track_name': 'count'
        }).reset_index()
        
        feature_efficiency['skip_rate'] = feature_efficiency['is_skip'] / feature_efficiency['track_name']
        
        # Artist efficiency
        artist_efficiency = df.groupby('artist_name', observed=False).agg({
            'listening_efficiency': 'mean',
            'is_skip': 'sum',
            'track_name': 'count',
            'ms_played': 'sum'
        }).reset_index()
        
        artist_efficiency['skip_rate'] = artist_efficiency['is_skip'] / artist_efficiency['track_name']
        artist_efficiency['listening_hours'] = artist_efficiency['ms_played'] / 3600000
        
        # Top and bottom artists by efficiency
        top_artists = artist_efficiency.nlargest(10, 'listening_efficiency')
        bottom_artists = artist_efficiency.nsmallest(10, 'listening_efficiency')
        
        results = {
            'efficiency_stats': efficiency_stats,
            'time_efficiency': time_efficiency,
            'feature_efficiency': feature_efficiency,
            'artist_efficiency': artist_efficiency,
            'top_efficient_artists': top_artists.to_dict('records'),
            'bottom_efficient_artists': bottom_artists.to_dict('records'),
            'overall_efficiency': df['listening_efficiency'].mean(),
            'overall_skip_rate': df['is_skip'].mean()
        }
        
        self.results['listening_efficiency'] = results
        logger.info("Listening efficiency analysis completed")
        
        return results
    
    def generate_insights(self) -> Dict:
        """
        Generate comprehensive insights from all analyses
        
        Returns:
            Dictionary with key insights
        """
        insights = {
            'temporal_insights': self._generate_temporal_insights(),
            'session_insights': self._generate_session_insights(),
            'platform_insights': self._generate_platform_insights(),
            'efficiency_insights': self._generate_efficiency_insights()
        }
        
        return insights
    
    def _generate_temporal_insights(self) -> List[str]:
        """Generate insights from temporal analysis"""
        if 'temporal_patterns' not in self.results:
            return []
        
        insights = []
        temp = self.results['temporal_patterns']
        
        # Daily patterns
        if 'daily_patterns' in temp:
            peak_day = temp['daily_patterns']['peak_day']
            insights.append(f"Peak listening day: {peak_day['day_of_week']} with {peak_day['listening_hours']:.1f} hours")
        
        # Seasonal patterns
        if 'seasonal_patterns' in temp:
            favorite_season = temp['seasonal_patterns']['favorite_season']
            insights.append(f"Favorite listening season: {favorite_season}")
        
        # Time of day
        if 'time_of_day_patterns' in temp:
            favorite_time = temp['time_of_day_patterns']['favorite_time']
            insights.append(f"Peak listening time: {favorite_time}")
        
        return insights
    
    def _generate_session_insights(self) -> List[str]:
        """Generate insights from session analysis"""
        if 'session_behavior' not in self.results:
            return []
        
        insights = []
        session = self.results['session_behavior']
        
        avg_duration = session['average_session_duration']
        insights.append(f"Average session duration: {avg_duration:.1f} minutes")
        
        binge_info = session['binge_sessions']
        insights.append(f"Binge listening sessions: {binge_info['count']} ({binge_info['percentage']:.1f}%)")
        
        return insights
    
    def _generate_platform_insights(self) -> List[str]:
        """Generate insights from platform analysis"""
        if 'platform_usage' not in self.results:
            return []
        
        insights = []
        platform = self.results['platform_usage']
        
        most_used = platform['most_used_platform']
        insights.append(f"Most used platform: {most_used}")
        
        return insights
    
    def _generate_efficiency_insights(self) -> List[str]:
        """Generate insights from efficiency analysis"""
        if 'listening_efficiency' not in self.results:
            return []
        
        insights = []
        efficiency = self.results['listening_efficiency']
        
        overall_eff = efficiency['overall_efficiency']
        skip_rate = efficiency['overall_skip_rate']
        
        insights.append(f"Overall listening efficiency: {overall_eff:.1%}")
        insights.append(f"Overall skip rate: {skip_rate:.1%}")
        
        return insights


def main():
    """Main function for testing the analyzer"""
    analyzer = ListeningPatternsAnalyzer()
    logger.info("Listening patterns analyzer ready for use")


if __name__ == "__main__":
    main()
