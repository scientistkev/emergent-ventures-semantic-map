"""
Data cleaning and normalization utilities for Emergent Ventures dataset.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from rapidfuzz import fuzz, process


def normalize_domain(domain: str, domain_mapping: Dict[str, str]) -> str:
    """
    Normalize a domain string using fuzzy matching against known domains.
    
    Args:
        domain: Domain string to normalize
        domain_mapping: Dictionary mapping variations to canonical forms
        
    Returns:
        Normalized domain string
    """
    domain_lower = domain.lower().strip()
    
    # Direct match
    if domain_lower in domain_mapping:
        return domain_mapping[domain_lower]
    
    # Fuzzy match against known domains
    if domain_mapping:
        best_match = process.extractOne(
            domain_lower,
            domain_mapping.keys(),
            scorer=fuzz.ratio,
            score_cutoff=85
        )
        if best_match:
            return domain_mapping[best_match[0]]
    
    # Return normalized version (lowercase, stripped)
    return domain_lower


def normalize_domains(domains: List[str], domain_mapping: Dict[str, str]) -> List[str]:
    """
    Normalize a list of domain strings.
    
    Args:
        domains: List of domain strings
        domain_mapping: Dictionary mapping variations to canonical forms
        
    Returns:
        List of normalized, unique domains
    """
    if not domains:
        return []
    
    normalized = [normalize_domain(d, domain_mapping) for d in domains]
    # Remove duplicates while preserving order
    seen = set()
    unique_normalized = []
    for d in normalized:
        if d not in seen:
            seen.add(d)
            unique_normalized.append(d)
    
    return unique_normalized


def normalize_location(location: str) -> str:
    """
    Normalize location string.
    
    Args:
        location: Location string (may contain multiple locations separated by '/')
        
    Returns:
        Normalized location string
    """
    if not location:
        return ""
    
    # Handle multi-location entries
    locations = [loc.strip() for loc in str(location).split('/')]
    # For now, keep primary location (first one)
    # Could be enhanced to handle multiple locations properly
    return locations[0] if locations else ""


def enhance_embedding_text(entry: Dict[str, Any]) -> str:
    """
    Create enhanced embedding text from entry data.
    
    Format: "{name} works on {project_name}: {project_description}. Domains: {domains}. Category: {category}."
    
    Args:
        entry: Dictionary containing entry data
        
    Returns:
        Enhanced embedding text string
    """
    name = entry.get('name', '')
    project_name = entry.get('project_name', '')
    project_desc = entry.get('project_description', '')
    domains = entry.get('domains', [])
    category = entry.get('category', '')
    
    # Build embedding text
    parts = []
    
    if name:
        parts.append(name)
    
    if project_name:
        parts.append(f"works on {project_name}")
    
    if project_desc:
        parts.append(f": {project_desc}")
    
    if domains:
        domains_str = ", ".join(domains)
        parts.append(f"Domains: {domains_str}")
    
    if category:
        parts.append(f"Category: {category}")
    
    embedding_text = ". ".join(parts) + "."
    
    return embedding_text


def create_domain_mapping(all_domains: List[str]) -> Dict[str, str]:
    """
    Create a mapping from domain variations to canonical forms.
    
    Args:
        all_domains: List of all domain strings found in dataset
        
    Returns:
        Dictionary mapping variations to canonical forms
    """
    # Define canonical domain mappings
    canonical_mappings = {
        'ai': 'artificial intelligence',
        'artificial intelligence': 'artificial intelligence',
        'machine learning': 'artificial intelligence',
        'ml': 'artificial intelligence',
        'healthtech': 'healthcare',
        'health tech': 'healthcare',
        'medical devices': 'healthcare',
        'medical': 'healthcare',
        'biotech': 'biotechnology',
        'biotechnology': 'biotechnology',
        'regenerative medicine': 'biotechnology',
        'stem cells': 'biotechnology',
        'neurology': 'healthcare',
        'agritech': 'agriculture',
        'agriculture': 'agriculture',
        'hardware': 'hardware',
        'software': 'software',
        'robotics': 'robotics',
        'automation': 'automation',
        'energy': 'energy',
        'evs': 'energy',
        'battery technology': 'energy',
        'transportation': 'transportation',
        'education': 'education',
        'fintech': 'financial technology',
        'civic technology': 'civic technology',
        'legal tech': 'legal technology',
        'open-source': 'open source',
        'open source': 'open source',
        'platforms': 'platforms',
        'physics': 'physics',
        'mathematics': 'mathematics',
        'category theory': 'mathematics',
        'theoretical physics': 'physics',
        'complex systems': 'complex systems',
        'social modeling': 'social modeling',
        'sustainability': 'sustainability',
        'materials science': 'materials science',
        'climate adaptation': 'climate adaptation',
        'aerospace': 'aerospace',
        'community building': 'community building',
        'stem': 'education',
        'talent development': 'education',
        'career development': 'career development',
        'engineering': 'engineering',
        'wearables': 'hardware',
        'labor safety': 'safety',
        'digital access': 'education',
        'smes': 'business',
        'business education': 'education',
        'learning platforms': 'education',
        'assistive technology': 'healthcare',
        'hr tech': 'human resources',
        'human resources': 'human resources',
    }
    
    # Build mapping from all domains to canonical forms
    domain_mapping = {}
    for domain in all_domains:
        domain_lower = domain.lower().strip()
        # Find canonical form
        canonical = canonical_mappings.get(domain_lower, domain_lower)
        domain_mapping[domain_lower] = canonical
    
    return domain_mapping


def clean_entry(entry: Dict[str, Any], domain_mapping: Dict[str, str]) -> Dict[str, Any]:
    """
    Clean and normalize a single entry.
    
    Args:
        entry: Raw entry dictionary
        domain_mapping: Domain normalization mapping
        
    Returns:
        Cleaned entry dictionary with normalized fields
    """
    cleaned = entry.copy()
    
    # Normalize domains
    if 'domains' in cleaned and cleaned['domains']:
        cleaned['domains_normalized'] = normalize_domains(cleaned['domains'], domain_mapping)
    else:
        cleaned['domains_normalized'] = []
    
    # Normalize location
    if 'location' in cleaned:
        cleaned['location_normalized'] = normalize_location(cleaned['location'])
    else:
        cleaned['location_normalized'] = ""
    
    # Normalize category (already seems consistent, but ensure lowercase)
    if 'category' in cleaned and cleaned['category']:
        cleaned['category_normalized'] = cleaned['category'].lower().strip()
    else:
        cleaned['category_normalized'] = ""
    
    # Enhance embedding text
    cleaned['embedding_text'] = enhance_embedding_text(cleaned)
    
    return cleaned


def clean_dataset(data: List[Dict[str, Any]], vocab_dir: Path) -> tuple:
    """
    Clean entire dataset and create controlled vocabularies.
    
    Args:
        data: List of raw entry dictionaries
        vocab_dir: Directory to save vocabulary files
        
    Returns:
        Tuple of (cleaned_data, vocabularies_dict)
    """
    import pandas as pd
    
    # Collect all unique domains
    all_domains = []
    for entry in data:
        if 'domains' in entry and entry['domains']:
            all_domains.extend([d.lower().strip() for d in entry['domains']])
    
    # Create domain mapping
    domain_mapping = create_domain_mapping(list(set(all_domains)))
    
    # Collect all categories
    all_categories = set()
    for entry in data:
        if 'category' in entry and entry['category']:
            all_categories.add(entry['category'].lower().strip())
    
    # Collect all locations
    all_locations = set()
    for entry in data:
        if 'location' in entry and entry['location']:
            loc = str(entry['location']).split('/')[0].strip()
            if loc:
                all_locations.add(loc)
    
    # Clean all entries
    cleaned_data = [clean_entry(entry, domain_mapping) for entry in data]
    
    # Create vocabularies
    vocabularies = {
        'domains': sorted(set(domain_mapping.values())),
        'categories': sorted(all_categories),
        'locations': sorted(all_locations),
        'domain_mapping': domain_mapping
    }
    
    # Save vocabularies
    vocab_dir.mkdir(parents=True, exist_ok=True)
    
    with open(vocab_dir / 'domains.json', 'w') as f:
        json.dump(vocabularies['domains'], f, indent=2)
    
    with open(vocab_dir / 'categories.json', 'w') as f:
        json.dump(vocabularies['categories'], f, indent=2)
    
    with open(vocab_dir / 'locations.json', 'w') as f:
        json.dump(vocabularies['locations'], f, indent=2)
    
    with open(vocab_dir / 'domain_mapping.json', 'w') as f:
        json.dump(vocabularies['domain_mapping'], f, indent=2)
    
    return cleaned_data, vocabularies

