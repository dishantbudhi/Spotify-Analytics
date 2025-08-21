"""
Streaming History Parser for Spotify Personal Analytics Engine

This module handles the parsing and processing of Spotify's extended streaming history
from multiple JSON files spanning 2014-2025. It processes music, podcast, audiobook,
and video streaming data with comprehensive error handling and data validation.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timezone
import logging
from tqdm import tqdm
import re
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StreamingRecord:
    """Data class for standardized streaming records"""
    timestamp: datetime
    track_name: str
    artist_name: str
    album_name: Optional[str]
    track_uri: Optional[str]
    ms_played: int
    reason_start: Optional[str]
    reason_end: Optional[str]
    shuffle: Optional[bool]
    skipped: Optional[bool]
    offline: Optional[bool]
    offline_timestamp: Optional[datetime]
    incognito_mode: Optional[bool]
    platform: Optional[str]
    user_agent: Optional[str]
    ip_address: Optional[str]
    country: Optional[str]
    content_type: str  # 'music', 'podcast', 'audiobook', 'video'
    file_source: str


class StreamingHistoryParser:
    """
    Comprehensive parser for Spotify streaming history data
    
    Handles multiple file types and formats across 10+ years of data:
    - Music streaming history (22 files, 2014-2025)
    - Video streaming history
    - Podcast streaming history
    - Audiobook streaming history
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.processed_data = []
        self.stats = {
            'total_files': 0,
            'total_records': 0,
            'music_records': 0,
            'podcast_records': 0,
            'audiobook_records': 0,
            'video_records': 0,
            'errors': 0,
            'date_range': {'start': None, 'end': None}
        }
        
    def discover_files(self) -> Dict[str, List[Path]]:
        """
        Discover and categorize all streaming history files
        
        Returns:
            Dictionary categorizing files by type
        """
        files = {
            'music': [],
            'video': [],
            'podcast': [],
            'audiobook': [],
            'other': []
        }
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist")
            return files
            
        for file_path in self.data_dir.glob("*.json"):
            filename = file_path.name.lower()
            
            if "streaming_history_audio" in filename:
                files['music'].append(file_path)
            elif "streaming_history_video" in filename:
                files['video'].append(file_path)
            elif "streaminghistory_podcast" in filename:
                files['podcast'].append(file_path)
            elif "streaminghistory_audiobook" in filename:
                files['audiobook'].append(file_path)
            else:
                files['other'].append(file_path)
                
        # Sort files by name to ensure chronological order
        for category in files:
            files[category].sort(key=lambda x: x.name)
            
        logger.info(f"Discovered files: {sum(len(files[k]) for k in files)} total")
        for category, file_list in files.items():
            logger.info(f"  {category}: {len(file_list)} files")
            
        return files
    
    def parse_timestamp(self, ts_value) -> datetime:
        """
        Parse Spotify timestamp to datetime object
        
        Args:
            ts_value: Timestamp value from Spotify data (string or int)
            
        Returns:
            Parsed datetime object
        """
        try:
            # Handle None or empty values
            if not ts_value:
                return datetime.now(tz=timezone.utc)
            
            # Handle integer Unix timestamps (milliseconds or seconds)
            if isinstance(ts_value, (int, float)):
                # Check if it's in milliseconds (> 1e10) or seconds
                if ts_value > 1e10:
                    # Milliseconds
                    return datetime.fromtimestamp(ts_value / 1000, tz=timezone.utc)
                else:
                    # Seconds
                    return datetime.fromtimestamp(ts_value, tz=timezone.utc)
            
            # Handle string timestamps
            ts_str = str(ts_value).strip()
            if not ts_str:
                return datetime.now(tz=timezone.utc)
                
            # ISO format with Z suffix
            if ts_str.endswith('Z'):
                return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            # ISO format with timezone
            elif 'T' in ts_str and '+' in ts_str:
                return datetime.fromisoformat(ts_str)
            # Try parsing as Unix timestamp string
            else:
                timestamp = float(ts_str)
                if timestamp > 1e10:
                    # Milliseconds
                    return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
                else:
                    # Seconds
                    return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except Exception as e:
            logger.warning(f"Failed to parse timestamp {ts_value} (type: {type(ts_value)}): {e}")
            return datetime.now(tz=timezone.utc)
    
    def parse_music_file(self, file_path: Path) -> List[StreamingRecord]:
        """
        Parse music streaming history file
        
        Args:
            file_path: Path to music streaming history JSON file
            
        Returns:
            List of parsed streaming records
        """
        records = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                logger.warning(f"Expected list in {file_path}, got {type(data)}")
                return records
                
            for item in data:
                try:
                    record = StreamingRecord(
                        timestamp=self.parse_timestamp(item.get('ts', '')),
                        track_name=item.get('master_metadata_track_name', 'Unknown Track'),
                        artist_name=item.get('master_metadata_album_artist_name', 'Unknown Artist'),
                        album_name=item.get('master_metadata_album_album_name'),
                        track_uri=item.get('spotify_track_uri'),
                        ms_played=item.get('ms_played', 0),
                        reason_start=item.get('reason_start'),
                        reason_end=item.get('reason_end'),
                        shuffle=item.get('shuffle'),
                        skipped=item.get('skipped'),
                        offline=item.get('offline'),
                        offline_timestamp=self.parse_timestamp(item.get('offline_timestamp', '')) if item.get('offline_timestamp') else None,
                        incognito_mode=item.get('incognito_mode'),
                        platform=item.get('platform'),
                        user_agent=item.get('user_agent_decrypted'),
                        ip_address=item.get('ip_addr_decrypted'),
                        country=item.get('conn_country'),
                        content_type='music',
                        file_source=file_path.name
                    )
                    records.append(record)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse record in {file_path}: {e}")
                    self.stats['errors'] += 1
                    
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            self.stats['errors'] += 1
            
        return records
    
    def parse_podcast_file(self, file_path: Path) -> List[StreamingRecord]:
        """
        Parse podcast streaming history file
        
        Args:
            file_path: Path to podcast streaming history JSON file
            
        Returns:
            List of parsed streaming records
        """
        records = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                return records
                
            for item in data:
                try:
                    record = StreamingRecord(
                        timestamp=self.parse_timestamp(item.get('ts', '')),
                        track_name=item.get('episode_name', 'Unknown Episode'),
                        artist_name=item.get('show_name', 'Unknown Show'),
                        album_name=item.get('show_name'),
                        track_uri=item.get('spotify_episode_uri'),
                        ms_played=item.get('ms_played', 0),
                        reason_start=item.get('reason_start'),
                        reason_end=item.get('reason_end'),
                        shuffle=None,  # Not applicable for podcasts
                        skipped=item.get('skipped'),
                        offline=item.get('offline'),
                        offline_timestamp=self.parse_timestamp(item.get('offline_timestamp', '')) if item.get('offline_timestamp') else None,
                        incognito_mode=item.get('incognito_mode'),
                        platform=item.get('platform'),
                        user_agent=item.get('user_agent_decrypted'),
                        ip_address=item.get('ip_addr_decrypted'),
                        country=item.get('conn_country'),
                        content_type='podcast',
                        file_source=file_path.name
                    )
                    records.append(record)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse podcast record in {file_path}: {e}")
                    self.stats['errors'] += 1
                    
        except Exception as e:
            logger.error(f"Failed to read podcast file {file_path}: {e}")
            self.stats['errors'] += 1
            
        return records
    
    def parse_audiobook_file(self, file_path: Path) -> List[StreamingRecord]:
        """
        Parse audiobook streaming history file
        
        Args:
            file_path: Path to audiobook streaming history JSON file
            
        Returns:
            List of parsed streaming records
        """
        records = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                return records
                
            for item in data:
                try:
                    record = StreamingRecord(
                        timestamp=self.parse_timestamp(item.get('ts', '')),
                        track_name=item.get('chapter_name', 'Unknown Chapter'),
                        artist_name=item.get('book_name', 'Unknown Book'),
                        album_name=item.get('book_name'),
                        track_uri=item.get('spotify_episode_uri'),
                        ms_played=item.get('ms_played', 0),
                        reason_start=item.get('reason_start'),
                        reason_end=item.get('reason_end'),
                        shuffle=None,
                        skipped=item.get('skipped'),
                        offline=item.get('offline'),
                        offline_timestamp=self.parse_timestamp(item.get('offline_timestamp', '')) if item.get('offline_timestamp') else None,
                        incognito_mode=item.get('incognito_mode'),
                        platform=item.get('platform'),
                        user_agent=item.get('user_agent_decrypted'),
                        ip_address=item.get('ip_addr_decrypted'),
                        country=item.get('conn_country'),
                        content_type='audiobook',
                        file_source=file_path.name
                    )
                    records.append(record)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse audiobook record in {file_path}: {e}")
                    self.stats['errors'] += 1
                    
        except Exception as e:
            logger.error(f"Failed to read audiobook file {file_path}: {e}")
            self.stats['errors'] += 1
            
        return records
    
    def parse_video_file(self, file_path: Path) -> List[StreamingRecord]:
        """
        Parse video streaming history file
        
        Args:
            file_path: Path to video streaming history JSON file
            
        Returns:
            List of parsed streaming records
        """
        records = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                return records
                
            for item in data:
                try:
                    record = StreamingRecord(
                        timestamp=self.parse_timestamp(item.get('ts', '')),
                        track_name=item.get('video_name', 'Unknown Video'),
                        artist_name=item.get('channel_name', 'Unknown Channel'),
                        album_name=item.get('channel_name'),
                        track_uri=item.get('spotify_video_uri'),
                        ms_played=item.get('ms_played', 0),
                        reason_start=item.get('reason_start'),
                        reason_end=item.get('reason_end'),
                        shuffle=None,
                        skipped=item.get('skipped'),
                        offline=item.get('offline'),
                        offline_timestamp=self.parse_timestamp(item.get('offline_timestamp', '')) if item.get('offline_timestamp') else None,
                        incognito_mode=item.get('incognito_mode'),
                        platform=item.get('platform'),
                        user_agent=item.get('user_agent_decrypted'),
                        ip_address=item.get('ip_addr_decrypted'),
                        country=item.get('conn_country'),
                        content_type='video',
                        file_source=file_path.name
                    )
                    records.append(record)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse video record in {file_path}: {e}")
                    self.stats['errors'] += 1
                    
        except Exception as e:
            logger.error(f"Failed to read video file {file_path}: {e}")
            self.stats['errors'] += 1
            
        return records
    
    def parse_all_files(self, use_multiprocessing: bool = True) -> pd.DataFrame:
        """
        Parse all streaming history files and return consolidated DataFrame
        
        Args:
            use_multiprocessing: Whether to use multiprocessing for faster parsing
            
        Returns:
            Consolidated DataFrame with all streaming records
        """
        files = self.discover_files()
        all_records = []
        
        if use_multiprocessing:
            # Use ThreadPoolExecutor for I/O-bound file reading
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                
                # Submit music files
                for file_path in files['music']:
                    future = executor.submit(self.parse_music_file, file_path)
                    futures.append(('music', future))
                
                # Submit other file types
                for file_path in files['podcast']:
                    future = executor.submit(self.parse_podcast_file, file_path)
                    futures.append(('podcast', future))
                    
                for file_path in files['audiobook']:
                    future = executor.submit(self.parse_audiobook_file, file_path)
                    futures.append(('audiobook', future))
                    
                for file_path in files['video']:
                    future = executor.submit(self.parse_video_file, file_path)
                    futures.append(('video', future))
                
                # Collect results with progress bar
                for content_type, future in tqdm(futures, desc="Parsing files"):
                    try:
                        records = future.result()
                        all_records.extend(records)
                        self.stats[f'{content_type}_records'] += len(records)
                    except Exception as e:
                        logger.error(f"Failed to parse {content_type} files: {e}")
        else:
            # Sequential processing
            for file_path in tqdm(files['music'], desc="Parsing music files"):
                records = self.parse_music_file(file_path)
                all_records.extend(records)
                self.stats['music_records'] += len(records)
                
            for file_path in tqdm(files['podcast'], desc="Parsing podcast files"):
                records = self.parse_podcast_file(file_path)
                all_records.extend(records)
                self.stats['podcast_records'] += len(records)
                
            for file_path in tqdm(files['audiobook'], desc="Parsing audiobook files"):
                records = self.parse_audiobook_file(file_path)
                all_records.extend(records)
                self.stats['audiobook_records'] += len(records)
                
            for file_path in tqdm(files['video'], desc="Parsing video files"):
                records = self.parse_video_file(file_path)
                all_records.extend(records)
                self.stats['video_records'] += len(records)
        
        # Convert to DataFrame
        df = pd.DataFrame([vars(record) for record in all_records])
        
        # Update statistics
        self.stats['total_files'] = sum(len(files[k]) for k in files)
        self.stats['total_records'] = len(df)
        
        if len(df) > 0:
            self.stats['date_range']['start'] = df['timestamp'].min()
            self.stats['date_range']['end'] = df['timestamp'].max()
        
        logger.info(f"Parsing complete: {self.stats['total_records']} records from {self.stats['total_files']} files")
        logger.info(f"Date range: {self.stats['date_range']['start']} to {self.stats['date_range']['end']}")
        
        return df
    
    def get_parsing_stats(self) -> Dict:
        """
        Get comprehensive parsing statistics
        
        Returns:
            Dictionary with parsing statistics
        """
        return self.stats.copy()


def main():
    """Main function for testing the parser"""
    parser = StreamingHistoryParser()
    df = parser.parse_all_files()
    
    if len(df) > 0:
        print(f"\nParsing Results:")
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Content types: {df['content_type'].value_counts().to_dict()}")
        print(f"Top artists: {df['artist_name'].value_counts().head(10).to_dict()}")
        
        # Save processed data
        output_path = Path("data/processed/streaming_history_parsed.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"Saved processed data to {output_path}")
    else:
        print("No data found to parse")


if __name__ == "__main__":
    main()
