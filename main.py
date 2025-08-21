#!/usr/bin/env python3
"""
BeatScope: Spotify Analytics - Main Pipeline

This script orchestrates the complete analytics pipeline from raw Spotify data
to interactive dashboard, including data parsing, analysis, and visualization generation.

⚠️  SECURITY NOTICE: This project processes personal Spotify data.
    - All data processing happens locally
    - Personal data files are excluded from version control
    - No data is shared externally
    - Use only your own exported Spotify data
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd
import json

# Add src to path for local imports
sys.path.append(str(Path(__file__).parent / "src"))

# Local imports
from data_processing.streaming_parser import StreamingHistoryParser
from data_processing.data_enrichment import DataEnrichment
from analysis.listening_patterns import ListeningPatternsAnalyzer

# Configure logging
def setup_logging():
    """Setup logging with proper directory creation"""
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/spotify_analytics.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()


class BeatScopePipeline:
    """
    Complete pipeline for BeatScope: Spotify Analytics
    
    Orchestrates the entire workflow:
    1. Parse streaming history from JSON files
    2. Enrich data with derived metrics
    3. Perform comprehensive analysis
    4. Generate visualizations and insights
    5. Launch interactive dashboard
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the analytics pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.setup_directories()
        
        # Initialize components
        self.parser = StreamingHistoryParser()
        self.enrichment = DataEnrichment(config_path)
        self.analyzer = ListeningPatternsAnalyzer(config_path)
        
        # Pipeline state
        self.pipeline_state = {
            'parsing_complete': False,
            'data_enrichment_complete': False,
            'analysis_complete': False,
            'dashboard_ready': False
        }
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            "data/raw",
            "data/processed", 
            "data/enriched",
            "data/cache",
            "logs",
            "results"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def run_full_pipeline(self, launch_dashboard: bool = True):
        """
        Run the complete analytics pipeline
        
        Args:
            launch_dashboard: Launch the dashboard after completion
        """
        logger.info("Starting BeatScope: Spotify Analytics Pipeline")
        start_time = datetime.now()
        
        try:
            # Step 1: Parse streaming history
            logger.info("Step 1: Parsing streaming history...")
            streaming_df = self.parse_streaming_history()
            
            if streaming_df.empty:
                logger.error("No streaming data found. Please ensure JSON files are in data/raw/")
                return
            
            # Step 2: Data enrichment and feature engineering
            logger.info("Step 2: Creating derived metrics...")
            final_df = self.create_derived_metrics(streaming_df)
            
            # Step 3: Comprehensive analysis
            logger.info("Step 3: Performing analysis...")
            analysis_results = self.perform_analysis(final_df)
            
            # Step 4: Generate insights
            logger.info("Step 4: Generating insights...")
            self.generate_insights(analysis_results)
            
            # Step 5: Launch dashboard
            if launch_dashboard:
                logger.info("Step 5: Launching dashboard...")
                self.launch_dashboard()
            
            # Pipeline complete
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info(f"Pipeline completed successfully in {duration}")
            self.print_summary()
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def parse_streaming_history(self) -> pd.DataFrame:
        """Parse streaming history from JSON files"""
        try:
            df = self.parser.parse_all_files(use_multiprocessing=True)
            
            if not df.empty:
                # Save parsed data
                output_path = Path("data/processed/streaming_history_parsed.parquet")
                df.to_parquet(output_path, index=False)
                logger.info(f"Saved parsed data to {output_path}")
                
                # Print parsing statistics
                stats = self.parser.get_parsing_stats()
                logger.info(f"Parsing complete: {stats['total_records']} records from {stats['total_files']} files")
                
                self.pipeline_state['parsing_complete'] = True
                return df
            else:
                logger.warning("No data parsed from JSON files")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to parse streaming history: {e}")
            return pd.DataFrame()
    

    
    def create_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived metrics and features"""
        try:
            # Handle missing data
            df = self.enrichment.handle_missing_data(df)
            
            # Create derived metrics
            df = self.enrichment.create_derived_metrics(df)
            
            # Save enriched data
            self.enrichment.save_enriched_data(df)
            
            logger.info(f"Data enrichment complete: {len(df)} records processed")
            
            self.pipeline_state['data_enrichment_complete'] = True
            return df
            
        except Exception as e:
            logger.error(f"Failed to create derived metrics: {e}")
            return df
    
    def perform_analysis(self, df: pd.DataFrame) -> dict:
        """Perform comprehensive analysis"""
        try:
            results = {}
            
            # Temporal patterns analysis
            logger.info("Analyzing temporal patterns...")
            results['temporal_patterns'] = self.analyzer.analyze_temporal_patterns(df)
            
            # Session behavior analysis
            logger.info("Analyzing session behavior...")
            results['session_behavior'] = self.analyzer.analyze_session_behavior(df)
            
            # Platform usage analysis
            logger.info("Analyzing platform usage...")
            results['platform_usage'] = self.analyzer.analyze_platform_usage(df)
            
            # Listening efficiency analysis
            logger.info("Analyzing listening efficiency...")
            results['listening_efficiency'] = self.analyzer.analyze_listening_efficiency(df)
            
            # Save analysis results
            self.save_analysis_results(results)
            
            logger.info("Analysis complete")
            
            self.pipeline_state['analysis_complete'] = True
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform analysis: {e}")
            return {}
    
    def generate_insights(self, analysis_results: dict) -> dict:
        """Generate insights from analysis results"""
        try:
            insights = self.analyzer.generate_insights()
            
            # Save insights
            insights_file = Path("results/key_insights.json")
            with open(insights_file, 'w') as f:
                json.dump(insights, f, indent=2, default=str)
            
            logger.info(f"Insights saved to {insights_file}")
            
            # Print key insights
            self.print_insights(insights)
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return {}
    
    def launch_dashboard(self):
        """Launch the interactive dashboard"""
        try:
            logger.info("Dashboard ready. Run 'streamlit run src/visualization/interactive_dashboard.py' to launch.")
            self.pipeline_state['dashboard_ready'] = True
            
        except Exception as e:
            logger.error(f"Failed to prepare dashboard: {e}")
    
    def save_analysis_results(self, results: dict):
        """Save analysis results to files"""
        try:
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            # Save each analysis result
            for analysis_name, analysis_result in results.items():
                result_file = results_dir / f"{analysis_name}_results.json"
                with open(result_file, 'w') as f:
                    json.dump(analysis_result, f, indent=2, default=str)
                
                logger.info(f"Saved {analysis_name} results to {result_file}")
                
        except Exception as e:
            logger.error(f"Failed to save analysis results: {e}")
    
    def print_insights(self, insights: dict):
        """Print key insights to console"""
        print("\n" + "="*50)
        print("KEY INSIGHTS")
        print("="*50)
        
        for category, category_insights in insights.items():
            if category_insights:
                print(f"\n{category.upper().replace('_', ' ')}:")
                for insight in category_insights:
                    print(f"  • {insight}")
        
        print("\n" + "="*50)
    
    def print_summary(self):
        """Print pipeline summary"""
        print("\n" + "="*50)
        print("PIPELINE SUMMARY")
        print("="*50)
        
        for step, completed in self.pipeline_state.items():
            status = "COMPLETE" if completed else "NOT COMPLETE"
            print(f"{step.replace('_', ' ').title()}: {status}")
        
        print("\nGenerated Files:")
        print("  • data/processed/ - Parsed and enriched data")
        print("  • data/enriched/ - Final analysis-ready data")
        print("  • results/ - Analysis results and insights")
        print("  • logs/ - Pipeline execution logs")
        
        print("\nNext Steps:")
        print("  • Run 'streamlit run src/visualization/interactive_dashboard.py' to launch dashboard")
        print("  • Check results/ directory for detailed analysis")
        print("  • Review logs/spotify_analytics.log for execution details")
        
        print("="*50)


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="BeatScope: Spotify Analytics Pipeline")
    parser.add_argument(
        "--config", 
        default="config/config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Don't launch dashboard after completion"
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Only parse streaming history (skip analysis)"
    )
    parser.add_argument(
        "--enrich-only",
        action="store_true",
        help="Only enrich data (skip analysis)"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = BeatScopePipeline(args.config)
    
    if args.parse_only:
        logger.info("Running parse-only mode...")
        df = pipeline.parse_streaming_history()
        if not df.empty:
            logger.info(f"Parsed {len(df)} records successfully")
    elif args.enrich_only:
        logger.info("Running enrich-only mode...")
        # Load existing parsed data
        parsed_file = Path("data/processed/streaming_history_parsed.parquet")
        if parsed_file.exists():
            df = pd.read_parquet(parsed_file)
            final_df = pipeline.create_derived_metrics(df)
            logger.info(f"Enriched {len(final_df)} records successfully")
        else:
            logger.error("No parsed data found. Run parse-only mode first.")
    else:
        # Run full pipeline
        pipeline.run_full_pipeline(
            launch_dashboard=not args.no_dashboard
        )


if __name__ == "__main__":
    main()
