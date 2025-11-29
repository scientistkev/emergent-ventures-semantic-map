"""
Change detection module to identify new/updated entries.
"""

from enum import Enum
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import hashlib


class ChangeType(str, Enum):
    """Types of changes detected."""
    NEW = "new"
    UPDATED = "updated"
    UNCHANGED = "unchanged"


@dataclass
class ChangeRecord:
    """Record of a detected change."""
    entry: Dict[str, Any]
    change_type: ChangeType
    hash_value: str
    previous_hash: Optional[str] = None
    differences: Optional[List[str]] = None


class ChangeDetector:
    """
    Detect changes between existing dataset and new scraped/extracted data.
    
    Identifies:
    - New entries (not in existing dataset)
    - Updated entries (changed from existing)
    - Unchanged entries
    """
    
    def __init__(self, hash_fields: Optional[List[str]] = None):
        """
        Initialize change detector.
        
        Args:
            hash_fields: Fields to include in hash for change detection.
                        If None, uses default fields.
        """
        self.hash_fields = hash_fields or [
            'name',
            'project_name',
            'project_description',
            'domains',
            'location',
            'cohort'
        ]
    
    def _compute_hash(self, entry: Dict[str, Any]) -> str:
        """
        Compute hash for an entry based on key fields.
        
        Args:
            entry: Entry dictionary
            
        Returns:
            Hash string
        """
        # Build hash string from key fields
        hash_parts = []
        for field in self.hash_fields:
            value = entry.get(field)
            if value is not None:
                if isinstance(value, list):
                    value = sorted([str(v) for v in value])
                    hash_parts.append(f"{field}:{','.join(value)}")
                else:
                    hash_parts.append(f"{field}:{str(value)}")
        
        hash_string = '|'.join(hash_parts)
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def _find_matching_entry(
        self,
        new_entry: Dict[str, Any],
        existing_entries: List[Dict[str, Any]]
    ) -> Optional[Tuple[Dict[str, Any], float]]:
        """
        Find matching entry in existing dataset.
        
        Uses name matching as primary key, with fuzzy matching.
        
        Args:
            new_entry: New entry to match
            existing_entries: List of existing entries
            
        Returns:
            Tuple of (matching_entry, similarity_score) or None
        """
        from rapidfuzz import fuzz
        
        new_name = str(new_entry.get('name', '')).lower().strip()
        if not new_name:
            return None
        
        best_match = None
        best_score = 0.0
        
        for existing in existing_entries:
            existing_name = str(existing.get('name', '')).lower().strip()
            if not existing_name:
                continue
            
            # Exact match
            if new_name == existing_name:
                return (existing, 1.0)
            
            # Fuzzy match
            score = fuzz.ratio(new_name, existing_name) / 100.0
            if score > best_score and score > 0.85:  # 85% similarity threshold
                best_score = score
                best_match = existing
        
        if best_match and best_score > 0.85:
            return (best_match, best_score)
        
        return None
    
    def _compare_entries(
        self,
        entry1: Dict[str, Any],
        entry2: Dict[str, Any]
    ) -> List[str]:
        """
        Compare two entries and return list of differences.
        
        Args:
            entry1: First entry
            entry2: Second entry
            
        Returns:
            List of difference descriptions
        """
        differences = []
        
        for field in self.hash_fields:
            val1 = entry1.get(field)
            val2 = entry2.get(field)
            
            if val1 != val2:
                differences.append(
                    f"{field}: '{val1}' -> '{val2}'"
                )
        
        return differences
    
    def detect_changes(
        self,
        new_entries: List[Dict[str, Any]],
        existing_entries: List[Dict[str, Any]]
    ) -> List[ChangeRecord]:
        """
        Detect changes between new and existing entries.
        
        Args:
            new_entries: List of new/updated entries
            existing_entries: List of existing entries
            
        Returns:
            List of ChangeRecord objects
        """
        change_records = []
        processed_existing_ids = set()
        
        for new_entry in new_entries:
            new_hash = self._compute_hash(new_entry)
            
            # Try to find matching entry
            match_result = self._find_matching_entry(new_entry, existing_entries)
            
            if match_result is None:
                # New entry
                change_records.append(ChangeRecord(
                    entry=new_entry,
                    change_type=ChangeType.NEW,
                    hash_value=new_hash
                ))
            else:
                existing_entry, similarity = match_result
                existing_hash = self._compute_hash(existing_entry)
                existing_id = id(existing_entry)
                processed_existing_ids.add(existing_id)
                
                if new_hash == existing_hash:
                    # Unchanged
                    change_records.append(ChangeRecord(
                        entry=new_entry,
                        change_type=ChangeType.UNCHANGED,
                        hash_value=new_hash,
                        previous_hash=existing_hash
                    ))
                else:
                    # Updated
                    differences = self._compare_entries(existing_entry, new_entry)
                    change_records.append(ChangeRecord(
                        entry=new_entry,
                        change_type=ChangeType.UPDATED,
                        hash_value=new_hash,
                        previous_hash=existing_hash,
                        differences=differences
                    ))
        
        return change_records
    
    def get_statistics(
        self,
        change_records: List[ChangeRecord]
    ) -> Dict[str, int]:
        """
        Get statistics about detected changes.
        
        Args:
            change_records: List of change records
            
        Returns:
            Dictionary with counts by change type
        """
        stats = {
            'new': 0,
            'updated': 0,
            'unchanged': 0,
            'total': len(change_records)
        }
        
        for record in change_records:
            stats[record.change_type.value] += 1
        
        return stats

