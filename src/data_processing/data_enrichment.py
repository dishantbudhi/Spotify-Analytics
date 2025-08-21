"""
Data Enrichment Module for Spotify Personal Analytics Engine

This module handles the merging and enrichment of streaming history data with
audio features, creating derived metrics and preparing data for analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataEnrichment:
    """
    Comprehensive data enrichment pipeline for Spotify analytics
    
    Features:
    - Merge streaming data with audio features
    - Create derived metrics and features
    - Handle missing data and data quality issues
    - Prepare data for analysis and visualization
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize data enrichment pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.analysis_config = self.config.get('analysis', {})
        
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
    
    def merge_streaming_and_audio_features(self, streaming_df: pd.DataFrame, 
                                         audio_features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge streaming history with audio features
        
        Args:
            streaming_df: DataFrame with streaming history
            audio_features_df: DataFrame with audio features
            
        Returns:
            Merged DataFrame
        """
        logger.info("Merging streaming data with audio features...")
        
        # Ensure track_uri is available for merging
        if 'track_uri' not in streaming_df.columns:
            logger.error("track_uri column not found in streaming data")
            return streaming_df
        
        if 'track_id' not in audio_features_df.columns:
            logger.error("track_id column not found in audio features data")
            return streaming_df
        
        # Merge on track URI/ID
        merged_df = streaming_df.merge(
            audio_features_df,
            left_on='track_uri',
            right_on='track_id',
            how='left',
            suffixes=('', '_audio')
        )
        
        logger.info(f"Merged {len(merged_df)} records")
        logger.info(f"Records with audio features: {merged_df['track_id'].notna().sum()}")
        
        return merged_df
    
    def create_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived metrics and features for analysis
        
        Args:
            df: Input DataFrame with streaming and audio data
            
        Returns:
            DataFrame with additional derived metrics
        """
        logger.info("Creating derived metrics...")
        
        df = df.copy()
        
        # Time-based features
        df = self._add_temporal_features(df)
        
        # Listening behavior features
        df = self._add_listening_behavior_features(df)
        
        # Audio feature derived metrics
        df = self._add_audio_feature_metrics(df)
        
        # Session and context features
        df = self._add_session_features(df)
        
        # Content type specific features
        df = self._add_content_type_features(df)
        
        logger.info("Derived metrics created successfully")
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        # Extract datetime components
        df['date'] = df['timestamp'].dt.date  # Add date column for daily analysis
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        df['quarter'] = df['timestamp'].dt.quarter
        
        # Time of day categories
        df['time_of_day'] = pd.cut(
            df['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            include_lowest=True
        )
        
        # Day type
        df['day_type'] = df['day_of_week'].map({
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
            4: 'Friday', 5: 'Weekend', 6: 'Weekend'
        })
        
        # Season
        df['season'] = pd.cut(
            df['month'],
            bins=[0, 3, 6, 9, 12],
            labels=['Winter', 'Spring', 'Summer', 'Fall'],
            include_lowest=True
        )
        
        return df
    
    def _add_listening_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add listening behavior derived features"""
        min_listening_duration = self.analysis_config.get('min_listening_duration', 30000)
        skip_threshold = self.analysis_config.get('skip_threshold', 0.3)
        
        # Listening duration in minutes
        df['listening_duration_minutes'] = df['ms_played'] / 60000
        
        # Skip detection
        df['is_skip'] = df['skipped'] == True
        
        # Handle missing duration_ms
        if 'duration_ms' in df.columns:
            df['skip_by_duration'] = (df['ms_played'] / df['duration_ms']) < skip_threshold
        else:
            # Estimate skip based on ms_played threshold when duration is unknown
            df['skip_by_duration'] = df['ms_played'] < 30000  # 30 seconds threshold
        
        # Complete listen detection
        df['is_complete_listen'] = (
            (df['ms_played'] >= min_listening_duration) & 
            (df['skip_by_duration'] == False)
        )
        
        # Listening efficiency (percentage of track played)
        if 'duration_ms' in df.columns:
            df['listening_efficiency'] = np.where(
                df['duration_ms'] > 0,
                df['ms_played'] / df['duration_ms'],
                0
            )
        else:
            # Estimate efficiency based on ms_played when duration is unknown
            # Assume average song length of 3 minutes (180000 ms)
            df['listening_efficiency'] = np.where(
                df['ms_played'] > 0,
                np.minimum(df['ms_played'] / 180000, 1.0),  # Cap at 1.0
                0
            )
        
        # Platform categories
        df['platform_category'] = df['platform'].fillna('Unknown').str.lower()
        df['platform_category'] = df['platform_category'].map({
            'android': 'Mobile',
            'ios': 'Mobile',
            'web_player': 'Web',
            'desktop': 'Desktop',
            'unknown': 'Unknown'
        }).fillna('Other')
        
        return df
    
    def _add_audio_feature_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived audio feature metrics"""
        # Check if audio features are available
        audio_features = ['valence', 'energy', 'danceability', 'tempo', 
                         'instrumentalness', 'speechiness', 'acousticness', 'liveness']
        
        missing_features = [f for f in audio_features if f not in df.columns]
        
        if missing_features:
            logger.warning(f"Missing audio features: {missing_features}. Using default values.")
            
            # Add default columns for missing features
            for feature in missing_features:
                df[feature] = 0.5  # Neutral default value
        
        # Mood classification based on valence and energy
        df['mood'] = self._classify_mood(df['valence'], df['energy'])
        
        # Energy level classification
        df['energy_level'] = pd.cut(
            df['energy'],
            bins=[0, 0.33, 0.66, 1.0],
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
        
        # Danceability level
        df['danceability_level'] = pd.cut(
            df['danceability'],
            bins=[0, 0.33, 0.66, 1.0],
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
        
        # Tempo categories (handle missing tempo)
        if 'tempo' in df.columns and not df['tempo'].isna().all():
            df['tempo_category'] = pd.cut(
                df['tempo'],
                bins=[0, 60, 90, 120, 150, 200, 300],
                labels=['Larghissimo', 'Largo', 'Moderato', 'Allegro', 'Vivace', 'Presto'],
                include_lowest=True
            )
        else:
            df['tempo_category'] = 'Unknown'
        
        # Musical complexity (combination of features)
        df['musical_complexity'] = (
            df['instrumentalness'] * 0.3 +
            df['speechiness'] * 0.2 +
            (1 - df['acousticness']) * 0.3 +
            df['liveness'] * 0.2
        )
        
        # Genre inference (simplified)
        df['inferred_genre'] = self._infer_genre_from_features(df)
        
        return df
    
    def _add_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add session and context features"""
        session_timeout = self.analysis_config.get('session_timeout', 1800000)  # 30 minutes
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Session detection
        df['time_since_last'] = df.groupby('content_type')['timestamp'].diff().dt.total_seconds() * 1000
        df['new_session'] = (
            (df['time_since_last'] > session_timeout) | 
            (df['time_since_last'].isna())
        )
        df['session_id'] = df.groupby('content_type')['new_session'].cumsum()
        
        # Session statistics
        session_stats = df.groupby(['content_type', 'session_id'], observed=False).agg({
            'ms_played': 'sum',
            'track_name': 'count',
            'timestamp': ['min', 'max']
        }).reset_index()
        
        session_stats.columns = ['content_type', 'session_id', 'session_duration', 'tracks_in_session', 'session_start', 'session_end']
        session_stats['session_duration_minutes'] = session_stats['session_duration'] / 60000
        
        # Merge session stats back
        df = df.merge(session_stats, on=['content_type', 'session_id'], how='left')
        
        return df
    
    def _add_content_type_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add content type specific features"""
        # Content type distribution
        df['is_music'] = df['content_type'] == 'music'
        df['is_podcast'] = df['content_type'] == 'podcast'
        df['is_audiobook'] = df['content_type'] == 'audiobook'
        df['is_video'] = df['content_type'] == 'video'
        
        # Content type specific metrics
        df['content_duration_hours'] = df['ms_played'] / 3600000
        
        return df
    
    def _classify_mood(self, valence: pd.Series, energy: pd.Series) -> pd.Series:
        """Classify mood based on valence and energy"""
        mood_conditions = [
            (valence >= 0.5) & (energy >= 0.5),
            (valence >= 0.5) & (energy < 0.5),
            (valence < 0.5) & (energy >= 0.5),
            (valence < 0.5) & (energy < 0.5)
        ]
        mood_choices = ['Happy', 'Calm', 'Energetic', 'Sad']
        
        return np.select(mood_conditions, mood_choices, default='Neutral')
    
    def _infer_genre_from_features(self, df: pd.DataFrame) -> pd.Series:
        """Infer genre from audio features (simplified)"""
        # Simple genre inference based on audio features
        genre_conditions = [
            (df['danceability'] > 0.7) & (df['energy'] > 0.7),
            (df['acousticness'] > 0.7) & (df['instrumentalness'] < 0.3),
            (df['instrumentalness'] > 0.7),
            (df['speechiness'] > 0.5),
            (df['energy'] > 0.8) & (df['tempo'] > 140),
            (df['valence'] < 0.3) & (df['energy'] < 0.5)
        ]
        genre_choices = ['Pop/Dance', 'Folk/Acoustic', 'Instrumental', 'Hip-Hop/Rap', 'Electronic', 'Sad/Emo']
        
        return np.select(genre_conditions, genre_choices, default='Other')
    
    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data and data quality issues
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing data handled
        """
        logger.info("Handling missing data...")
        
        df = df.copy()
        
        # Fill missing values
        df['track_name'] = df['track_name'].fillna('Unknown Track')
        df['artist_name'] = df['artist_name'].fillna('Unknown Artist')
        df['album_name'] = df['album_name'].fillna('Unknown Album')
        
        # Audio features missing values
        audio_features = [
            'danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'
        ]
        
        for feature in audio_features:
            if feature in df.columns:
                if feature in ['key', 'mode', 'time_signature']:
                    df[feature] = df[feature].fillna(-1)  # -1 for unknown
                else:
                    df[feature] = df[feature].fillna(df[feature].median())
        
        # Boolean features
        boolean_features = ['shuffle', 'skipped', 'offline', 'incognito_mode']
        for feature in boolean_features:
            if feature in df.columns:
                df[feature] = df[feature].astype('object').fillna(False)
        
        # Categorical features
        categorical_features = ['platform', 'user_agent', 'ip_address', 'country']
        for feature in categorical_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna('Unknown')
        
        logger.info("Missing data handling completed")
        return df
    
    def create_aggregated_metrics(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create aggregated metrics for different time periods and dimensions
        
        Args:
            df: Enriched DataFrame
            
        Returns:
            Dictionary of aggregated DataFrames
        """
        logger.info("Creating aggregated metrics...")
        
        aggregated_data = {}
        
        # Daily aggregation
        daily_stats = df.groupby(['date', 'content_type'], observed=False).agg({
            'ms_played': 'sum',
            'track_name': 'count',
            'is_complete_listen': 'sum',
            'is_skip': 'sum',
            'listening_efficiency': 'mean',
            'valence': 'mean',
            'energy': 'mean',
            'danceability': 'mean',
            'tempo': 'mean'
        }).reset_index()
        
        daily_stats['listening_hours'] = daily_stats['ms_played'] / 3600000
        daily_stats['skip_rate'] = daily_stats['is_skip'] / daily_stats['track_name']
        daily_stats['completion_rate'] = daily_stats['is_complete_listen'] / daily_stats['track_name']
        
        aggregated_data['daily_stats'] = daily_stats
        
        # Weekly aggregation
        weekly_stats = df.groupby(['year', 'week_of_year', 'content_type'], observed=False).agg({
            'ms_played': 'sum',
            'track_name': 'count',
            'artist_name': 'nunique',
            'album_name': 'nunique',
            'valence': 'mean',
            'energy': 'mean',
            'danceability': 'mean',
            'tempo': 'mean'
        }).reset_index()
        
        weekly_stats['listening_hours'] = weekly_stats['ms_played'] / 3600000
        weekly_stats['unique_artists'] = weekly_stats['artist_name']
        weekly_stats['unique_albums'] = weekly_stats['album_name']
        
        aggregated_data['weekly_stats'] = weekly_stats
        
        # Monthly aggregation
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
        
        aggregated_data['monthly_stats'] = monthly_stats
        
        # Artist aggregation
        artist_stats = df.groupby('artist_name', observed=False).agg({
            'ms_played': 'sum',
            'track_name': 'count',
            'album_name': 'nunique',
            'valence': 'mean',
            'energy': 'mean',
            'danceability': 'mean',
            'tempo': 'mean',
            'listening_efficiency': 'mean',
            'is_complete_listen': 'sum',
            'is_skip': 'sum'
        }).reset_index()
        
        artist_stats['listening_hours'] = artist_stats['ms_played'] / 3600000
        artist_stats['unique_albums'] = artist_stats['album_name']
        artist_stats['skip_rate'] = artist_stats['is_skip'] / artist_stats['track_name']
        artist_stats['completion_rate'] = artist_stats['is_complete_listen'] / artist_stats['track_name']
        
        aggregated_data['artist_stats'] = artist_stats
        
        # Time of day aggregation
        time_stats = df.groupby(['time_of_day', 'content_type'], observed=False).agg({
            'ms_played': 'sum',
            'track_name': 'count',
            'valence': 'mean',
            'energy': 'mean',
            'danceability': 'mean',
            'tempo': 'mean'
        }).reset_index()
        
        time_stats['listening_hours'] = time_stats['ms_played'] / 3600000
        
        aggregated_data['time_stats'] = time_stats
        
        logger.info("Aggregated metrics created successfully")
        return aggregated_data
    
    def save_enriched_data(self, df: pd.DataFrame, output_dir: str = "data/enriched"):
        """
        Save enriched data to files
        
        Args:
            df: Enriched DataFrame
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main enriched dataset
        main_file = output_path / "enriched_streaming_data.parquet"
        df.to_parquet(main_file, index=False)
        logger.info(f"Saved enriched data to {main_file}")
        
        # Save aggregated metrics
        aggregated_data = self.create_aggregated_metrics(df)
        
        for name, agg_df in aggregated_data.items():
            agg_file = output_path / f"{name}.parquet"
            agg_df.to_parquet(agg_file, index=False)
            logger.info(f"Saved {name} to {agg_file}")
        
        # Save summary statistics
        summary_stats = self._create_summary_statistics(df)
        summary_file = output_path / "summary_statistics.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        logger.info(f"Saved summary statistics to {summary_file}")
    
    def _create_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Create comprehensive summary statistics"""
        summary = {
            'dataset_overview': {
                'total_records': len(df),
                'date_range': {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat()
                },
                'content_types': df['content_type'].value_counts().to_dict(),
                'unique_tracks': df['track_name'].nunique(),
                'unique_artists': df['artist_name'].nunique(),
                'unique_albums': df['album_name'].nunique()
            },
            'listening_behavior': {
                'total_listening_hours': df['ms_played'].sum() / 3600000,
                'average_session_duration': df['session_duration_minutes'].mean(),
                'skip_rate': df['is_skip'].mean(),
                'completion_rate': df['is_complete_listen'].mean(),
                'average_listening_efficiency': df['listening_efficiency'].mean()
            },
            'audio_features_summary': {
                'average_valence': df['valence'].mean(),
                'average_energy': df['energy'].mean(),
                'average_danceability': df['danceability'].mean(),
                'average_tempo': df['tempo'].mean(),
                'mood_distribution': df['mood'].value_counts().to_dict()
            },
            'platform_usage': {
                'platform_distribution': df['platform_category'].value_counts().to_dict()
            }
        }
        
        return summary


def main():
    """Main function for testing data enrichment"""
    # This would typically be called after parsing
    logger.info("Data enrichment module ready for use")


if __name__ == "__main__":
    main()
