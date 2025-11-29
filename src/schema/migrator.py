"""
Schema migration utilities to convert between flat and graph formats.
"""

from typing import List, Dict, Any
from pathlib import Path
import json
from .definitions import KnowledgeGraph


def convert_flat_to_graph(entries: List[Dict[str, Any]]) -> List[KnowledgeGraph]:
    """
    Convert list of flat entries to knowledge graph format.
    
    Args:
        entries: List of flat entry dictionaries
        
    Returns:
        List of KnowledgeGraph instances
    """
    graphs = []
    for entry in entries:
        graph = KnowledgeGraph.from_flat_entry(entry)
        graphs.append(graph)
    return graphs


def convert_graph_to_flat(graphs: List[KnowledgeGraph]) -> List[Dict[str, Any]]:
    """
    Convert knowledge graphs back to flat format (for backward compatibility).
    
    Args:
        graphs: List of KnowledgeGraph instances
        
    Returns:
        List of flat entry dictionaries
    """
    flat_entries = []
    
    for graph in graphs:
        # Extract person node
        person_nodes = [n for n in graph.nodes if n.node_type.value == 'Person']
        if not person_nodes:
            continue
        
        person = person_nodes[0]
        entry = {
            'name': person.properties.get('name', ''),
            'age': person.properties.get('age'),
            'education': person.properties.get('education'),
            'location': person.properties.get('location'),
            'source_url': person.properties.get('source_url'),
        }
        
        # Extract project node
        project_edges = [
            e for e in graph.edges
            if e.edge_type.value == 'PERSON_TO_PROJECT' and e.source_id == person.id
        ]
        if project_edges:
            project_id = project_edges[0].target_id
            project_nodes = [n for n in graph.nodes if n.id == project_id]
            if project_nodes:
                project = project_nodes[0]
                entry['project_name'] = project.properties.get('name', '')
                entry['project_description'] = project.properties.get('description', '')
                entry['category'] = project.properties.get('category')
                entry['funding_type'] = project.properties.get('funding_type')
                entry['links'] = project.properties.get('links', [])
        
        # Extract domains
        domain_edges = [
            e for e in graph.edges
            if e.edge_type.value == 'PERSON_TO_DOMAIN' and e.source_id == person.id
        ]
        domains = []
        for edge in domain_edges:
            domain_nodes = [n for n in graph.nodes if n.id == edge.target_id]
            if domain_nodes:
                domains.append(domain_nodes[0].properties.get('name'))
        entry['domains'] = domains
        
        # Extract cohort
        cohort_edges = [
            e for e in graph.edges
            if e.edge_type.value == 'COHORT_TO_PERSON' and e.target_id == person.id
        ]
        if cohort_edges:
            cohort_id = cohort_edges[0].source_id
            cohort_nodes = [n for n in graph.nodes if n.id == cohort_id]
            if cohort_nodes:
                entry['cohort'] = cohort_nodes[0].properties.get('name', '')
        
        flat_entries.append(entry)
    
    return flat_entries

