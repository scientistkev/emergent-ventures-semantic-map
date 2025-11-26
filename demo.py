#!/usr/bin/env python3
"""
Demo script for Emergent Ventures Semantic Map Phase 1.

This script demonstrates the complete pipeline:
1. Load cleaned data and embeddings
2. Perform a semantic search
3. Display results

Usage:
    python demo.py [query]
"""

import json
import numpy as np
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from search import SemanticSearch, format_search_results


def main():
    """Run semantic search demo."""
    # Set up paths
    data_path = project_root / "data" / "processed" / "cleaned_data.json"
    embeddings_path = project_root / "data" / "processed" / "embeddings.npy"
    
    # Check if files exist
    if not data_path.exists():
        print(f"Error: {data_path} not found. Please run the data cleaning notebook first.")
        sys.exit(1)
    
    if not embeddings_path.exists():
        print(f"Error: {embeddings_path} not found. Please run the embeddings notebook first.")
        sys.exit(1)
    
    # Load data and embeddings
    print("Loading data and embeddings...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    embeddings = np.load(embeddings_path)
    
    print(f"Loaded {len(data)} entries with {embeddings.shape[1]}-dimensional embeddings\n")
    
    # Initialize search engine
    search_engine = SemanticSearch(data, embeddings)
    
    # Get query from command line or use default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "AI and machine learning projects"
    
    print(f"Searching for: '{query}'")
    print("=" * 80)
    
    # Perform search
    results = search_engine.search(query, top_k=5)
    
    # Display results
    print(format_search_results(results))
    
    print("\n" + "=" * 80)
    print("Try different queries:")
    print("  python demo.py 'Healthcare and medical devices'")
    print("  python demo.py 'Hardware and robotics'")
    print("  python demo.py 'Education and learning platforms'")


if __name__ == "__main__":
    main()

