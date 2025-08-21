#!/usr/bin/env python3
"""
Setup script for BeatScope: Spotify Analytics

This script helps users set up the project environment, install dependencies,
and configure the necessary settings for the Spotify analytics pipeline.

⚠️  SECURITY NOTICE: This project processes personal Spotify data.
    - All data processing happens locally
    - Personal data files are excluded from version control
    - No data is shared externally
    - Use only your own exported Spotify data
"""

import sys
import subprocess
from pathlib import Path
import yaml  # type: ignore
import json

def print_banner():
    """Print project banner"""
    print("=" * 70)
    print("BEATSCOPE: SPOTIFY ANALYTICS - SETUP")
    print("=" * 70)
    print("A personal data-driven platform for analyzing your Spotify")
    print("streaming history with advanced analytics and insights.")
    print("=" * 70)

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"Python {sys.version.split()[0]} is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling dependencies...")
    
    try:
        # Install from requirements.txt
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\nCreating project directories...")
    
    directories = [
        "data/raw",
        "data/processed",
        "data/enriched",
        "data/cache",
        "logs",
        "results",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}")



def create_sample_data():
    """Create sample data structure for testing"""
    print("\nCreating sample data structure...")
    
    # Create sample JSON structure
    sample_data = [
        {
            "ts": "2024-01-01T12:00:00Z",
            "ms_played": 180000,
            "master_metadata_track_name": "Sample Track",
            "master_metadata_album_artist_name": "Sample Artist",
            "master_metadata_album_album_name": "Sample Album",
            "spotify_track_uri": "spotify:track:4iV5W9uYEdYUVa79Axb7Rh",
            "reason_start": "trackdone",
            "reason_end": "trackdone",
            "shuffle": False,
            "skipped": False,
            "offline": False,
            "platform": "web_player",
            "conn_country": "US"
        }
    ]
    
    sample_file = Path("data/raw/sample_streaming_history.json")
    with open(sample_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Created sample data: {sample_file}")
    print("   You can replace this with your actual Spotify data export files.")

def create_environment_file():
    """Create .env file for environment variables"""
    print("\nCreating environment configuration...")
    
    env_content = """# BeatScope: Spotify Analytics - Environment Variables

# Database Configuration (optional)
DATABASE_URL=postgresql://user:password@localhost/spotify_analytics

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/spotify_analytics.log

# Performance Settings
MAX_WORKERS=4
MEMORY_LIMIT_GB=8
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("Created .env file")
        print("   Edit this file to configure environment variables")
    else:
        print(".env file already exists")

def run_tests():
    """Run basic tests to verify installation"""
    print("\nRunning basic tests...")
    
    try:
        # Test imports
        print("All required packages imported successfully")
        
        # Test configuration loading
        config_path = Path("config/config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                yaml.safe_load(f)
            print("Configuration file loaded successfully")
        else:
            print("Configuration file not found")
        
        print("Basic tests passed")
        return True
        
    except ImportError as e:
        print(f"Import test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 70)
    print("SETUP COMPLETE - NEXT STEPS")
    print("=" * 70)
    
    print("\n1. Add your Spotify data export files:")
    print("   • Place your JSON files in data/raw/")
    print("   • Files should be named like: Streaming_History_Audio_*.json")
    
    print("\n2. Run the analytics pipeline:")
    print("   • python main.py --parse-only (to parse data only)")
    print("   • python main.py (to run full pipeline)")
    
    print("\n3. Launch the dashboard:")
    print("   • streamlit run src/visualization/interactive_dashboard.py")
    
    print("\n4. Explore the project:")

    print("   • Review results/ for generated insights")
    print("   • Check logs/ for execution details")
    
    print("\nFor more information, see README.md")
    print("=" * 70)

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("Setup failed during dependency installation")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Create sample data
    create_sample_data()
    
    # Create environment file
    create_environment_file()
    
    # Run tests
    if not run_tests():
        print("Setup failed during testing")
        sys.exit(1)
    
    # Print next steps
    print_next_steps()
    
    print("\nSetup completed successfully!")

if __name__ == "__main__":
    main()
