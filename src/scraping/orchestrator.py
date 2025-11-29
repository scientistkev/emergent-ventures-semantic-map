"""
Scraper orchestrator to coordinate multiple scrapers and manage data flow.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from .base import BaseScraper, ScrapedEntry
from .marginal_revolution import MarginalRevolutionScraper
from .mercatus import MercatusScraper
from .web import WebScraper


class ScraperOrchestrator:
    """
    Orchestrates multiple scrapers and manages data flow.
    
    Coordinates:
    - Multiple scrapers for different sources
    - Data flow: scrape → extract → validate → merge
    - Source attribution tracking
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize orchestrator.
        
        Args:
            output_dir: Directory to save scraped results
        """
        self.output_dir = output_dir or Path('data/raw/scraped')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize scrapers
        self.scrapers: Dict[str, BaseScraper] = {
            'marginal_revolution': MarginalRevolutionScraper(),
            'mercatus': MercatusScraper(),
            'web': WebScraper()
        }
        
        # Track processed sources
        self.processed_sources = []
    
    def determine_scraper_type(self, source: str) -> str:
        """
        Determine which scraper to use based on source.
        
        Args:
            source: Source URL or file path
            
        Returns:
            Scraper type key
        """
        source_lower = source.lower()
        
        if 'marginalrevolution.com' in source_lower or 'marginalrevolution.org' in source_lower:
            return 'marginal_revolution'
        elif source.endswith('.pdf') or 'mercatus.org' in source_lower:
            return 'mercatus'
        else:
            return 'web'
    
    def scrape_source(
        self,
        source: str,
        scraper_type: Optional[str] = None,
        save_results: bool = True
    ) -> List[ScrapedEntry]:
        """
        Scrape a single source.
        
        Args:
            source: Source URL or file path
            scraper_type: Type of scraper to use (auto-detected if None)
            save_results: Whether to save results to file
            
        Returns:
            List of scraped entries
        """
        # Determine scraper type
        if scraper_type is None:
            scraper_type = self.determine_scraper_type(source)
        
        if scraper_type not in self.scrapers:
            print(f"Unknown scraper type: {scraper_type}")
            return []
        
        scraper = self.scrapers[scraper_type]
        
        print(f"Scraping {source} using {scraper_type} scraper...")
        
        # Scrape
        entries = scraper.scrape(source)
        
        # Validate
        valid_entries = [e for e in entries if scraper.validate_entry(e)]
        invalid_count = len(entries) - len(valid_entries)
        
        if invalid_count > 0:
            print(f"Filtered out {invalid_count} invalid entries")
        
        print(f"Scraped {len(valid_entries)} valid entries from {source}")
        
        # Save results
        if save_results and valid_entries:
            # Create filename from source
            import re
            from urllib.parse import urlparse
            
            if source.startswith('http'):
                parsed = urlparse(source)
                filename = re.sub(r'[^\w\-_\.]', '_', parsed.path.replace('/', '_'))
                if not filename or filename == '_':
                    filename = parsed.netloc.replace('.', '_')
            else:
                filename = Path(source).stem
            
            output_path = self.output_dir / f"{scraper_type}_{filename}.json"
            scraper.save_results(valid_entries, output_path)
        
        # Track processed source
        self.processed_sources.append({
            'source': source,
            'scraper_type': scraper_type,
            'entry_count': len(valid_entries),
            'timestamp': __import__('time').time()
        })
        
        return valid_entries
    
    def scrape_multiple_sources(
        self,
        sources: List[str],
        save_results: bool = True
    ) -> List[ScrapedEntry]:
        """
        Scrape multiple sources.
        
        Args:
            sources: List of source URLs or file paths
            save_results: Whether to save results to files
            
        Returns:
            Combined list of all scraped entries
        """
        all_entries = []
        
        for source in sources:
            try:
                entries = self.scrape_source(source, save_results=save_results)
                all_entries.extend(entries)
            except Exception as e:
                print(f"Error scraping {source}: {e}")
                continue
        
        return all_entries
    
    def merge_entries(self, entries: List[ScrapedEntry]) -> List[Dict[str, Any]]:
        """
        Merge scraped entries into a unified format.
        
        Args:
            entries: List of scraped entries
            
        Returns:
            List of merged entry dictionaries
        """
        merged = []
        
        for entry in entries:
            merged_entry = {
                'source_url': entry.source_url,
                'cohort': entry.cohort,
                'raw_content': entry.raw_content,
                'metadata': entry.metadata,
                'extracted_data': entry.extracted_data,
                'timestamp': entry.timestamp
            }
            merged.append(merged_entry)
        
        return merged
    
    def save_merged_results(
        self,
        entries: List[ScrapedEntry],
        output_path: Path
    ):
        """
        Save merged entries to a single JSON file.
        
        Args:
            entries: List of scraped entries
            output_path: Path to save merged JSON file
        """
        merged = self.merge_entries(entries)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(merged)} merged entries to {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about scraped data.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_sources': len(self.processed_sources),
            'sources_by_type': {},
            'total_entries': 0
        }
        
        for source_info in self.processed_sources:
            scraper_type = source_info['scraper_type']
            entry_count = source_info['entry_count']
            
            if scraper_type not in stats['sources_by_type']:
                stats['sources_by_type'][scraper_type] = {
                    'count': 0,
                    'entries': 0
                }
            
            stats['sources_by_type'][scraper_type]['count'] += 1
            stats['sources_by_type'][scraper_type]['entries'] += entry_count
            stats['total_entries'] += entry_count
        
        return stats

