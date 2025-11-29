#!/usr/bin/env python3
"""
Example script demonstrating Phase 2 pipeline usage.

This script shows how to use the Phase 2 scraping, extraction, and pipeline components.
"""

import sys
from pathlib import Path

# Add project root to path (not just src)
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now use absolute imports from src package
from src.pipeline.run import Pipeline
from src.config import Config
from src.scraping.orchestrator import ScraperOrchestrator


def example_config():
    """Example: Create and use configuration."""
    print("=" * 80)
    print("Example 1: Configuration Management")
    print("=" * 80)
    
    # Create config
    config = Config.create_default_config()
    
    # Add a source
    config.add_source(
        name="EV India 13 Announcement",
        url="https://marginalrevolution.com/marginalrevolution/2024/01/emergent-ventures-india-13.html",
        source_type="marginal_revolution",
        enabled=True,
        notes="Example Marginal Revolution blog post"
    )
    
    print(f"Created config with {len(config.sources)} source(s)")
    print(f"Enabled sources: {len(config.get_enabled_sources())}")


def example_scraping():
    """Example: Scraping from sources."""
    print("\n" + "=" * 80)
    print("Example 2: Scraping")
    print("=" * 80)
    
    orchestrator = ScraperOrchestrator()
    
    # Example: Scrape a single source (would need actual URL)
    # sources = ["https://example.com/ev-cohort"]
    # entries = orchestrator.scrape_multiple_sources(sources)
    # print(f"Scraped {len(entries)} entries")
    
    print("Note: Uncomment actual URLs to test scraping")


def example_pipeline():
    """Example: Full pipeline execution."""
    print("\n" + "=" * 80)
    print("Example 3: Full Pipeline")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = Pipeline(
        data_dir=Path("data"),
        skip_embeddings=True  # Skip for faster testing
    )
    
    print("Pipeline initialized")
    print(f"Data directory: {pipeline.data_dir}")
    print(f"Cleaned data path: {pipeline.cleaned_data_path}")
    
    # Example: Run full pipeline (would need actual sources)
    # sources = [
    #     "https://marginalrevolution.com/marginalrevolution/2024/01/ev-india-13.html",
    # ]
    # 
    # stats = pipeline.run_full_pipeline(
    #     sources=sources,
    #     incremental=True,
    #     apply_labeling=True
    # )
    # 
    # print(f"Pipeline results: {stats}")
    
    print("Note: Uncomment and provide actual source URLs to run full pipeline")


def main():
    """Run examples."""
    print("Phase 2 Pipeline Examples")
    print("=" * 80)
    print()
    
    try:
        example_config()
        example_scraping()
        example_pipeline()
        
        print("\n" + "=" * 80)
        print("Examples complete!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Add actual source URLs to config or pipeline")
        print("2. Run scraping to collect data")
        print("3. Run full pipeline to process and extract")
        print("4. Check results in data/processed/cleaned_data.json")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

