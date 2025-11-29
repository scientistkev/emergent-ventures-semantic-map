"""
End-to-end pipeline orchestration: scrape → extract → clean → embed → cluster
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from ..scraping.orchestrator import ScraperOrchestrator
from ..scraping.base import ScrapedEntry
from ..extraction.llm_extractor import LLMExtractor
from ..extraction.parser import parse_extraction_result, validate_extracted_data
from ..extraction.labeling import AutoLabeler
from ..data_cleaning import clean_dataset
from ..embeddings import generate_embeddings_for_dataset
from .detector import ChangeDetector, ChangeType


class Pipeline:
    """
    End-to-end pipeline for processing EV winner data.
    
    Orchestrates:
    - Scraping from multiple sources
    - LLM extraction of structured data
    - Data cleaning and normalization
    - Embedding generation
    - Change detection
    - Incremental updates
    """
    
    def __init__(
        self,
        data_dir: Path,
        vocab_dir: Optional[Path] = None,
        skip_embeddings: bool = False
    ):
        """
        Initialize pipeline.
        
        Args:
            data_dir: Base data directory
            vocab_dir: Vocabulary directory (defaults to data_dir/vocabularies)
            skip_embeddings: If True, skip embedding generation (faster for testing)
        """
        self.data_dir = Path(data_dir)
        self.vocab_dir = vocab_dir or self.data_dir / 'vocabularies'
        self.skip_embeddings = skip_embeddings
        
        # Initialize components
        self.scraper_orchestrator = ScraperOrchestrator(
            output_dir=self.data_dir / 'raw' / 'scraped'
        )
        self.extractor = LLMExtractor()
        self.labeler = AutoLabeler(vocab_dir=self.vocab_dir)
        self.change_detector = ChangeDetector()
        
        # Paths
        self.raw_data_path = self.data_dir / 'raw' / 'data.json'
        self.cleaned_data_path = self.data_dir / 'processed' / 'cleaned_data.json'
        self.embeddings_path = self.data_dir / 'processed' / 'embeddings.npy'
    
    def load_existing_data(self) -> List[Dict[str, Any]]:
        """Load existing cleaned data."""
        if self.cleaned_data_path.exists():
            with open(self.cleaned_data_path, 'r') as f:
                return json.load(f)
        return []
    
    def scrape_sources(
        self,
        sources: List[str],
        save_individual: bool = True
    ) -> List[ScrapedEntry]:
        """
        Scrape data from sources.
        
        Args:
            sources: List of source URLs or file paths
            save_individual: Whether to save individual source results
            
        Returns:
            List of scraped entries
        """
        print(f"Scraping {len(sources)} sources...")
        entries = self.scraper_orchestrator.scrape_multiple_sources(
            sources,
            save_results=save_individual
        )
        print(f"Scraped {len(entries)} entries total")
        return entries
    
    def extract_structured_data(
        self,
        scraped_entries: List[ScrapedEntry],
        apply_labeling: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract structured data from scraped entries.
        
        Args:
            scraped_entries: List of scraped entries
            apply_labeling: Whether to apply automated labeling
            
        Returns:
            List of extracted and structured entries
        """
        print(f"Extracting structured data from {len(scraped_entries)} entries...")
        
        extracted_entries = []
        
        for entry in tqdm(scraped_entries, desc="Extracting"):
            # Extract using LLM
            extraction_result = self.extractor.extract(
                entry.raw_content,
                cohort=entry.cohort
            )
            
            # Parse and validate
            parsed_data, warnings = parse_extraction_result(extraction_result)
            
            if not parsed_data:
                print(f"Skipping entry due to validation errors: {warnings}")
                continue
            
            # Add source metadata
            parsed_data['source_url'] = entry.source_url
            parsed_data['cohort'] = parsed_data.get('cohort') or entry.cohort
            parsed_data['raw_content'] = entry.raw_content
            parsed_data['_source_metadata'] = entry.metadata
            
            # Apply automated labeling if requested
            if apply_labeling:
                parsed_data = self.labeler.label_entry(parsed_data)
            
            extracted_entries.append(parsed_data)
        
        print(f"Extracted {len(extracted_entries)} valid entries")
        return extracted_entries
    
    def clean_and_normalize(
        self,
        extracted_entries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Clean and normalize extracted entries.
        
        Args:
            extracted_entries: List of extracted entries
            
        Returns:
            List of cleaned entries
        """
        print("Cleaning and normalizing data...")
        
        # Use existing cleaning pipeline
        cleaned_data, vocabularies = clean_dataset(
            extracted_entries,
            self.vocab_dir
        )
        
        print(f"Cleaned {len(cleaned_data)} entries")
        return cleaned_data
    
    def detect_changes(
        self,
        new_entries: List[Dict[str, Any]],
        existing_entries: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect changes between new and existing entries.
        
        Args:
            new_entries: Newly processed entries
            existing_entries: Existing entries (loaded if None)
            
        Returns:
            Dictionary with 'new', 'updated', 'unchanged' lists
        """
        if existing_entries is None:
            existing_entries = self.load_existing_data()
        
        print(f"Detecting changes ({len(new_entries)} new, {len(existing_entries)} existing)...")
        
        change_records = self.change_detector.detect_changes(
            new_entries,
            existing_entries
        )
        
        stats = self.change_detector.get_statistics(change_records)
        print(f"Changes detected: {stats['new']} new, {stats['updated']} updated, {stats['unchanged']} unchanged")
        
        # Group by change type
        grouped = {
            'new': [],
            'updated': [],
            'unchanged': []
        }
        
        for record in change_records:
            grouped[record.change_type.value].append(record.entry)
        
        return grouped
   
    def merge_entries(
        self,
        existing_entries: List[Dict[str, Any]],
        changes: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Merge new/updated entries with existing entries.
        
        Args:
            existing_entries: Existing entries
            changes: Dictionary of changes by type
            
        Returns:
            Merged list of all entries
        """
        # Create lookup by name for updates
        existing_lookup = {e.get('name'): e for e in existing_entries if e.get('name')}
        
        merged = []
        processed_names = set()
        
        # Add unchanged entries
        for entry in changes['unchanged']:
            name = entry.get('name')
            if name and name not in processed_names:
                merged.append(entry)
                processed_names.add(name)
        
        # Add/update with new and updated entries
        for entry in changes['new'] + changes['updated']:
            name = entry.get('name')
            if name:
                merged.append(entry)
                processed_names.add(name)
        
        # Add remaining existing entries that weren't in new data
        for entry in existing_entries:
            name = entry.get('name')
            if name and name not in processed_names:
                merged.append(entry)
                processed_names.add(name)
        
        return merged
    
    def generate_embeddings(
        self,
        cleaned_entries: List[Dict[str, Any]]
    ):
        """
        Generate embeddings for cleaned entries.
        
        Args:
            cleaned_entries: List of cleaned entries
        """
        if self.skip_embeddings:
            print("Skipping embedding generation (skip_embeddings=True)")
            return
        
        print("Generating embeddings...")
        generate_embeddings_for_dataset(
            cleaned_entries,
            self.embeddings_path,
            text_field='embedding_text'
        )
    
    def run_full_pipeline(
        self,
        sources: List[str],
        incremental: bool = True,
        apply_labeling: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Args:
            sources: List of source URLs or file paths to scrape
            incremental: If True, only process new/updated entries
            apply_labeling: Whether to apply automated labeling
            
        Returns:
            Dictionary with pipeline results and statistics
        """
        print("=" * 80)
        print("Starting full pipeline")
        print("=" * 80)
        
        # Step 1: Scrape
        scraped_entries = self.scrape_sources(sources)
        
        if not scraped_entries:
            print("No entries scraped. Exiting.")
            return {'error': 'No entries scraped'}
        
        # Step 2: Extract
        extracted_entries = self.extract_structured_data(
            scraped_entries,
            apply_labeling=apply_labeling
        )
        
        if not extracted_entries:
            print("No valid entries extracted. Exiting.")
            return {'error': 'No valid entries extracted'}
        
        # Step 3: Clean
        cleaned_entries = self.clean_and_normalize(extracted_entries)
        
        # Step 4: Detect changes (if incremental)
        existing_entries = None
        changes = None
        if incremental:
            existing_entries = self.load_existing_data()
            changes = self.detect_changes(cleaned_entries, existing_entries)
            
            # Merge entries
            all_entries = self.merge_entries(existing_entries, changes)
        else:
            all_entries = cleaned_entries
        
        # Step 5: Save cleaned data
        self.cleaned_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cleaned_data_path, 'w') as f:
            json.dump(all_entries, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(all_entries)} entries to {self.cleaned_data_path}")
        
        # Step 6: Generate embeddings
        self.generate_embeddings(all_entries)
        
        # Return statistics
        stats = {
            'scraped': len(scraped_entries),
            'extracted': len(extracted_entries),
            'cleaned': len(cleaned_entries),
            'total_entries': len(all_entries),
            'changes': changes if changes else None
        }
        
        print("=" * 80)
        print("Pipeline complete!")
        print("=" * 80)
        
        return stats

