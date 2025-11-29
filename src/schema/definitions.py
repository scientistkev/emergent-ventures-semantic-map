"""
Knowledge-graph ready schema definitions.

Defines node types and edge types for future graph analytics,
while maintaining backward compatibility with flat JSON format.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


class NodeType(str, Enum):
    """Node types in the knowledge graph."""
    PERSON = "Person"
    PROJECT = "Project"
    DOMAIN = "Domain"
    GEOGRAPHY = "Geography"
    OUTCOME = "Outcome"
    INSTITUTION = "Institution"
    COHORT = "Cohort"


class EdgeType(str, Enum):
    """Edge types in the knowledge graph."""
    PERSON_TO_PROJECT = "PERSON_TO_PROJECT"
    PERSON_TO_DOMAIN = "PERSON_TO_DOMAIN"
    DOMAIN_TO_DOMAIN = "DOMAIN_TO_DOMAIN"
    PROJECT_TO_OUTCOME = "PROJECT_TO_OUTCOME"
    COHORT_TO_PERSON = "COHORT_TO_PERSON"
    PERSON_TO_GEOGRAPHY = "PERSON_TO_GEOGRAPHY"
    PERSON_TO_INSTITUTION = "PERSON_TO_INSTITUTION"
    PROJECT_TO_DOMAIN = "PROJECT_TO_DOMAIN"


@dataclass
class BaseNode:
    """Base node class for knowledge graph."""
    id: str
    node_type: NodeType
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PersonNode(BaseNode):
    """Person node representing an EV winner."""
    def __init__(
        self,
        id: str,
        name: str,
        age: Optional[int] = None,
        education: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            id=id,
            node_type=NodeType.PERSON,
            properties={
                'name': name,
                'age': age,
                'education': education,
                **(properties or {})
            }
        )
    
    @property
    def name(self) -> str:
        return self.properties['name']


@dataclass
class ProjectNode(BaseNode):
    """Project node representing a winner's project."""
    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        category: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            id=id,
            node_type=NodeType.PROJECT,
            properties={
                'name': name,
                'description': description,
                'category': category,
                **(properties or {})
            }
        )
    
    @property
    def name(self) -> str:
        return self.properties['name']


@dataclass
class DomainNode(BaseNode):
    """Domain node representing a field/domain."""
    def __init__(
        self,
        id: str,
        name: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            id=id,
            node_type=NodeType.DOMAIN,
            properties={
                'name': name,
                **(properties or {})
            }
        )
    
    @property
    def name(self) -> str:
        return self.properties['name']


@dataclass
class GeographyNode(BaseNode):
    """Geography node representing a location."""
    def __init__(
        self,
        id: str,
        name: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            id=id,
            node_type=NodeType.GEOGRAPHY,
            properties={
                'name': name,
                **(properties or {})
            }
        )
    
    @property
    def name(self) -> str:
        return self.properties['name']


@dataclass
class OutcomeNode(BaseNode):
    """Outcome node representing a project outcome."""
    def __init__(
        self,
        id: str,
        outcome_type: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            id=id,
            node_type=NodeType.OUTCOME,
            properties={
                'outcome_type': outcome_type,
                **(properties or {})
            }
        )
    
    @property
    def outcome_type(self) -> str:
        return self.properties['outcome_type']


@dataclass
class InstitutionNode(BaseNode):
    """Institution node representing an organization."""
    def __init__(
        self,
        id: str,
        name: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            id=id,
            node_type=NodeType.INSTITUTION,
            properties={
                'name': name,
                **(properties or {})
            }
        )
    
    @property
    def name(self) -> str:
        return self.properties['name']


@dataclass
class CohortNode(BaseNode):
    """Cohort node representing an EV cohort."""
    def __init__(
        self,
        id: str,
        name: str,
        date: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            id=id,
            node_type=NodeType.COHORT,
            properties={
                'name': name,
                'date': date,
                **(properties or {})
            }
        )
    
    @property
    def name(self) -> str:
        return self.properties['name']


@dataclass
class Edge:
    """Edge in the knowledge graph."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate edge type."""
        if not isinstance(self.edge_type, EdgeType):
            self.edge_type = EdgeType(self.edge_type)


@dataclass
class KnowledgeGraph:
    """Knowledge graph containing nodes and edges."""
    nodes: List[BaseNode] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    
    def add_node(self, node: BaseNode):
        """Add a node to the graph."""
        if not any(n.id == node.id for n in self.nodes):
            self.nodes.append(node)
    
    def add_edge(self, edge: Edge):
        """Add an edge to the graph."""
        # Validate that source and target nodes exist
        source_exists = any(n.id == edge.source_id for n in self.nodes)
        target_exists = any(n.id == edge.target_id for n in self.nodes)
        
        if not source_exists or not target_exists:
            raise ValueError(
                f"Edge references non-existent nodes: "
                f"source={edge.source_id}, target={edge.target_id}"
            )
        
        self.edges.append(edge)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary format."""
        return {
            'nodes': [
                {
                    'id': node.id,
                    'type': node.node_type.value,
                    'properties': node.properties
                }
                for node in self.nodes
            ],
            'edges': [
                {
                    'source': edge.source_id,
                    'target': edge.target_id,
                    'type': edge.edge_type.value,
                    'properties': edge.properties
                }
                for edge in self.edges
            ]
        }
    
    @classmethod
    def from_flat_entry(cls, entry: Dict[str, Any]) -> 'KnowledgeGraph':
        """
        Convert a flat entry to knowledge graph structure.
        
        This maintains backward compatibility while enabling future graph analytics.
        
        Args:
            entry: Flat entry dictionary (Phase 1 format)
            
        Returns:
            KnowledgeGraph instance
        """
        import hashlib
        
        graph = cls()
        
        # Generate IDs
        person_id = hashlib.md5(entry.get('name', '').encode()).hexdigest()[:12]
        project_id = hashlib.md5(
            (entry.get('project_name', '') + entry.get('name', '')).encode()
        ).hexdigest()[:12]
        cohort_id = hashlib.md5(entry.get('cohort', '').encode()).hexdigest()[:12]
        
        # Create person node
        person_node = PersonNode(
            id=person_id,
            name=entry.get('name', ''),
            age=entry.get('age'),
            education=entry.get('education'),
            properties={
                'location': entry.get('location'),
                'source_url': entry.get('source_url'),
                **entry.get('metadata', {})
            }
        )
        graph.add_node(person_node)
        
        # Create project node
        if entry.get('project_name') or entry.get('project_description'):
            project_node = ProjectNode(
                id=project_id,
                name=entry.get('project_name', 'Unnamed Project'),
                description=entry.get('project_description', ''),
                category=entry.get('category'),
                properties={
                    'funding_type': entry.get('funding_type'),
                    'links': entry.get('links', [])
                }
            )
            graph.add_node(project_node)
            
            # Add person -> project edge
            graph.add_edge(Edge(
                source_id=person_id,
                target_id=project_id,
                edge_type=EdgeType.PERSON_TO_PROJECT
            ))
        
        # Create domain nodes and edges
        domains = entry.get('domains', []) or entry.get('domains_normalized', [])
        for domain_name in domains:
            if domain_name:
                domain_id = hashlib.md5(domain_name.encode()).hexdigest()[:12]
                domain_node = DomainNode(id=domain_id, name=domain_name)
                graph.add_node(domain_node)
                
                # Add person -> domain edge
                graph.add_edge(Edge(
                    source_id=person_id,
                    target_id=domain_id,
                    edge_type=EdgeType.PERSON_TO_DOMAIN
                ))
                
                # Add project -> domain edge if project exists
                if entry.get('project_name'):
                    graph.add_edge(Edge(
                        source_id=project_id,
                        target_id=domain_id,
                        edge_type=EdgeType.PROJECT_TO_DOMAIN
                    ))
        
        # Create geography node and edge
        location = entry.get('location') or entry.get('location_normalized')
        if location:
            geo_id = hashlib.md5(location.encode()).hexdigest()[:12]
            geo_node = GeographyNode(id=geo_id, name=location)
            graph.add_node(geo_node)
            
            graph.add_edge(Edge(
                source_id=person_id,
                target_id=geo_id,
                edge_type=EdgeType.PERSON_TO_GEOGRAPHY
            ))
        
        # Create cohort node and edge
        cohort_name = entry.get('cohort')
        if cohort_name:
            cohort_node = CohortNode(
                id=cohort_id,
                name=cohort_name,
                date=entry.get('metadata', {}).get('date')
            )
            graph.add_node(cohort_node)
            
            graph.add_edge(Edge(
                source_id=cohort_id,
                target_id=person_id,
                edge_type=EdgeType.COHORT_TO_PERSON
            ))
        
        return graph

